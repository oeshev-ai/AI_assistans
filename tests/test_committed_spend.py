"""После job_committed расход пишется, даже если файл не дошёл."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import pytest
import respx

from app.infra.cost_tracker import cost_tracker, set_billing_user
from app.llm.assist_ai import AssistAIClient
from app.llm.errors import AssistAIError

_BASE = "https://polza.ai/api"
_MEDIA_URL = f"{_BASE}/v1/media"
_MODEL = "openai/gpt-image-2.6-flare"


@pytest.fixture(autouse=True)
async def _reset_spend(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[None]:
    monkeypatch.setenv("LLM_DAILY_BUDGET_USD", "10")
    monkeypatch.setenv("LLM_USER_DAILY_BUDGET_USD", "10")
    cost_tracker._memory.clear()
    await cost_tracker.release_reservation()
    set_billing_user(None)
    yield
    await cost_tracker.release_reservation()
    cost_tracker._memory.clear()
    set_billing_user(None)


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        recorded.append(seconds)

    monkeypatch.setattr("app.llm.polza_media._sleep", fake_sleep)
    monkeypatch.setattr("app.llm.image_ops._sleep", fake_sleep)
    monkeypatch.setattr("app.llm.transport._sleep", fake_sleep)
    return recorded


@pytest.mark.asyncio
async def test_poll_timeout_records_image_spend(
    multi_client_factory: Callable[..., AssistAIClient],
    polza_fixture: Callable[[str], dict[str, Any]],
    sleeps: list[float],
) -> None:
    pending = polza_fixture("async_pending")
    client = multi_client_factory(
        fallback="",
        polza_total_wait=8.0,
        polza_poll_interval=4.0,
    )
    set_billing_user(11)
    day = cost_tracker._day_key()
    before = float(cost_tracker._memory.get(f"cost:user:11:day:{day}", 0.0))

    with respx.mock(assert_all_called=False) as router:
        router.post(_MEDIA_URL).mock(return_value=httpx.Response(200, json=pending))
        router.get(f"{_BASE}/v1/media/{pending['id']}").mock(
            return_value=httpx.Response(200, json=pending)
        )
        with pytest.raises(AssistAIError) as info:
            await client.generate_image("постер", model=_MODEL, size="1024x1024")

    assert info.value.job_committed is True
    after = float(cost_tracker._memory.get(f"cost:user:11:day:{day}", 0.0))
    assert after > before
    assert sleeps
