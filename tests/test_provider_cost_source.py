"""G5/G6: источник цены (cost_rub > usage > table > flat) и метрики."""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest
import respx
import structlog
from prometheus_client import REGISTRY

from app.infra import cost_tracker as ct
from app.infra.cost_tracker import LLM_SPEND_USD, CostTracker
from app.infra.settings import get_settings
from app.llm.assist_ai import AssistAIClient
from tests.polza_helpers import make_polza_provider

_FLARE = "openai/gpt-image-2.6-flare"
_FLUX = "black-forest-labs/flux-3-image-klein"
_USAGE = {"input_tokens": 121, "output_tokens": 664}


def _sample(source: str, provider: str, model: str, kind: str = "image") -> float:
    return (
        REGISTRY.get_sample_value(
            "llm_cost_usd_total",
            {"kind": kind, "provider": provider, "model": model, "source": source},
        )
        or 0.0
    )


def test_cost_rub_has_priority_over_usage_and_table() -> None:
    estimate = CostTracker.estimate_image_cost_detailed(
        1,
        model=_FLARE,
        quality="high",
        size="1536x1024",
        usage=_USAGE,
        cost_rub=304.0,
        rub_per_usd=304.0,
    )
    assert estimate.source == "cost_rub"
    assert estimate.amount_usd == pytest.approx(1.0)
    assert estimate.cost_rub == 304.0


def test_cost_rub_from_usage_dict() -> None:
    estimate = CostTracker.estimate_image_cost_detailed(
        1, model=_FLARE, usage={**_USAGE, "cost_rub": 6.29}, rub_per_usd=304.0
    )
    assert estimate.source == "cost_rub"
    assert estimate.amount_usd == pytest.approx(6.29 / 304.0)


def test_polza_edit_is_about_6_3_rub_equivalent() -> None:
    estimate = CostTracker.estimate_image_cost_detailed(
        1,
        model=_FLARE,
        provider=make_polza_provider(),
        cost_rub=6.29,
        rub_per_usd=304.0,
    )
    assert estimate.amount_usd * 304.0 == pytest.approx(6.29)


def test_usage_tokens_then_table_then_flat() -> None:
    by_usage = CostTracker.estimate_image_cost_detailed(1, model=_FLARE, usage=_USAGE)
    assert by_usage.source == "usage"
    assert by_usage.amount_usd == pytest.approx(
        CostTracker.estimate_image_cost(1, model=_FLARE, usage=_USAGE)
    )
    by_table = CostTracker.estimate_image_cost_detailed(
        1, model=_FLARE, quality="high", size="1536x1024"
    )
    assert by_table.source == "table"
    flat = CostTracker.estimate_image_cost_detailed(
        1, model=_FLARE, quality="auto", size="auto"
    )
    assert flat.source == "flat"
    assert flat.amount_usd == pytest.approx(get_settings().cost_image_usd)


def test_multiplier_applies_to_non_rub_sources_only() -> None:
    polza = make_polza_provider(cost_multiplier=0.39)
    plain = CostTracker.estimate_image_cost(
        1, model=_FLARE, quality="high", size="1536x1024"
    )
    scaled = CostTracker.estimate_image_cost(
        1, model=_FLARE, quality="high", size="1536x1024", provider=polza
    )
    assert scaled == pytest.approx(plain * 0.39)
    assert CostTracker.estimate_image_cost(
        1, model=_FLARE, provider=polza, cost_rub=30.4, rub_per_usd=304.0
    ) == pytest.approx(0.1)


def test_non_openai_uses_flat_map_and_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COST_IMAGE_FLUX_USD", "0.04")
    monkeypatch.setenv("COST_IMAGE_FLAT_MAP", f'{{"{_FLUX}": 0.004}}')
    mapped = CostTracker.estimate_image_cost_detailed(1, model=_FLUX)
    assert mapped.source == "flat"
    assert mapped.amount_usd == pytest.approx(0.004)
    other = CostTracker.estimate_image_cost(1, model="recraft/recraft-v3")
    assert other == pytest.approx(0.04)
    monkeypatch.delenv("COST_IMAGE_FLAT_MAP")
    assert CostTracker.estimate_image_cost(1, model=_FLUX) == pytest.approx(0.04)


def test_flat_map_parsing() -> None:
    assert ct._parse_flat_map(None) == {}
    assert ct._parse_flat_map("{}") == {}
    assert ct._parse_flat_map("not json") == {}
    assert ct._parse_flat_map("[1]") == {}
    parsed = ct._parse_flat_map('{"A/B": 0.01, "bad": "x", "neg": -1}')
    assert parsed == {"a/b": 0.01}


@pytest.mark.asyncio
async def test_reserve_equals_fact_after_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_DAILY_BUDGET_USD", "100")
    tracker = CostTracker()
    polza = make_polza_provider()
    reserved = tracker.estimate_image_cost(
        1,
        model=_FLARE,
        quality="high",
        size="1536x1024",
        provider=polza,
        rub_per_usd=304.0,
    )
    fact = tracker.estimate_image_cost(
        1, model=_FLARE, provider=polza, cost_rub=6.29, rub_per_usd=304.0
    )
    # Оценка и факт на одной шкале (₽/304), расхождение в пределах ~40%.
    assert 0.6 <= reserved / fact <= 1.2
    await tracker.check_budget(estimated=reserved)
    day_key = f"cost:day:{tracker._day_key()}"
    assert tracker._memory[day_key] == pytest.approx(reserved)
    await tracker.record(
        kind="image", amount_usd=fact, provider="polza", source="cost_rub"
    )
    # Резерв снят, остался только факт.
    assert tracker._memory[day_key] == pytest.approx(fact)


@pytest.mark.asyncio
async def test_record_emits_new_metric_and_extended_log() -> None:
    tracker = CostTracker()
    model = "test/metric-model"
    before = _sample("cost_rub", "polza", model)
    with structlog.testing.capture_logs() as logs:
        await tracker.record(
            kind="image",
            amount_usd=0.5,
            provider="polza",
            model=model,
            source="cost_rub",
            cost_rub=152.0,
        )
    assert _sample("cost_rub", "polza", model) == pytest.approx(before + 0.5)
    event = next(item for item in logs if item["event"] == "llm_spend_recorded")
    assert event["provider"] == "polza"
    assert event["model"] == model
    assert event["source"] == "cost_rub"
    assert event["cost_rub"] == 152.0
    assert event["kind"] == "image"
    assert event["amount_usd"] == 0.5


def test_legacy_spend_metric_labels_unchanged() -> None:
    assert LLM_SPEND_USD._labelnames == ("kind",)


@pytest.mark.asyncio
async def test_client_records_cost_rub_source_for_polza(
    multi_client_factory: Callable[..., AssistAIClient],
) -> None:
    client = multi_client_factory()
    before = _sample("cost_rub", "polza", _FLARE)
    with respx.mock(assert_all_called=False) as router:
        router.post("https://polza.ai/api/v1/media").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "g1",
                    "status": "completed",
                    "data": [{"url": "https://s3.polza.ai/a.png"}],
                    "usage": {"cost_rub": 6.08, "cost": 6.08},
                },
            )
        )
        await client.generate_image("постер", model=_FLARE, size="1536x1024")
    assert _sample("cost_rub", "polza", _FLARE) == pytest.approx(before + 6.08 / 304.0)


@pytest.mark.asyncio
async def test_client_records_usage_source_for_proxyapi(
    assist_client: AssistAIClient,
) -> None:
    before = _sample("usage", "proxyapi", _FLARE)
    with respx.mock(assert_all_called=False) as router:
        router.post("https://api.proxyapi.ru/v1/images/generations").mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": [{"url": "https://cdn.example/a.png"}],
                    "usage": _USAGE,
                },
            )
        )
        await assist_client.generate_image("постер", model=_FLARE, size="1024x1024")
    assert _sample("usage", "proxyapi", _FLARE) > before


@pytest.mark.asyncio
async def test_client_falls_back_to_table_times_multiplier_without_cost_rub(
    multi_client_factory: Callable[..., AssistAIClient],
) -> None:
    client = multi_client_factory()
    before = _sample("usage", "polza", _FLARE)
    with respx.mock(assert_all_called=False) as router:
        router.post("https://polza.ai/api/v1/media").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "g2",
                    "status": "completed",
                    "data": [{"url": "https://s3.polza.ai/b.png"}],
                    "usage": {"input_tokens": 121, "output_tokens": 664},
                },
            )
        )
        await client.generate_image("постер", model=_FLARE, size="1536x1024")
    expected = CostTracker.estimate_image_cost(
        1,
        model=_FLARE,
        usage={"input_tokens": 121, "output_tokens": 664},
        provider=make_polza_provider(),
    )
    assert _sample("usage", "polza", _FLARE) == pytest.approx(before + expected)
