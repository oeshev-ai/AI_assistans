"""G3/G7/G8: тело ``/v1/media`` и ответы на фикстурах G0."""

from __future__ import annotations

import base64
import io
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
import respx
import structlog

from app.infra.cost_tracker import set_billing_user
from app.llm.assist_ai import (
    AssistAIClient,
    AssistAIError,
    max_edit_images,
    prepare_edit_files,
    supports_multi_image,
)
from app.llm.polza_media import (
    POLZA_MAX_REFERENCE_IMAGES,
    build_media_body,
    generate,
    hash_user_id,
    image_resolution_tier,
    map_media_error,
    parse_media_payload,
    polza_size_fields,
)
from app.llm.providers import ProviderConfig
from tests.polza_helpers import make_polza_provider

_MEDIA_URL = "https://polza.ai/api/v1/media"
_MODEL = "openai/gpt-image-2.6-flare"


def _png(color: tuple[int, int, int], size: tuple[int, int] = (8, 8)) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def _body(**kwargs: Any) -> dict[str, Any]:
    return build_media_body(make_polza_provider(), model=_MODEL, prompt="x", **kwargs)


def test_body_has_no_complexity_background_or_image_resolution() -> None:
    body = _body(aspect_ratio="3:2", size="1536x1024", quality="high")
    media_input = body["input"]
    assert media_input["aspect_ratio"] == "3:2"
    assert media_input["size"] == "1536x1024"
    assert media_input["quality"] == "high"
    for forbidden in ("complexity", "background", "image_resolution"):
        assert forbidden not in media_input
        assert forbidden not in body


def test_quality_is_passed_through_as_is() -> None:
    for quality in ("low", "xhigh", "max"):
        assert _body(quality=quality)["input"]["quality"] == quality


def test_base64_references_keep_base_first() -> None:
    base, ref = _png((255, 0, 0)), _png((0, 0, 255))
    media_input = _body(images=[base, ref])["input"]
    assert [item["type"] for item in media_input["images"]] == ["base64", "base64"]
    assert base64.b64decode(media_input["images"][0]["data"]) == base
    assert base64.b64decode(media_input["images"][1]["data"]) == ref


def test_provider_object_pins_upstream_and_caps_price() -> None:
    body = _body()
    assert body["provider"] == {
        "only": ["OpenAI"],
        "allow_fallbacks": False,
        "max_price": {"image": 12.0},
    }
    assert "require_parameters" not in body["provider"]
    assert "data_residency" not in body["provider"]


def test_zero_price_cap_omits_max_price() -> None:
    provider = make_polza_provider(max_price_image_rub=0.0)
    body = build_media_body(provider, model=_MODEL, prompt="x")
    assert "max_price" not in body["provider"]
    assert body["provider"]["only"] == ["OpenAI"]


def test_user_is_hashed_and_bounded() -> None:
    hashed = hash_user_id(123456789)
    assert hashed is not None
    assert "123456789" not in hashed
    assert len(hashed) <= 256
    assert hash_user_id(None) is None
    assert _body(user=hashed)["user"] == hashed


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        ("1536x1024", {"aspect_ratio": "3:2", "size": "1536x1024"}),
        ("1024x1536", {"aspect_ratio": "2:3", "size": "1024x1536"}),
        ("1024x1024", {"aspect_ratio": "1:1", "size": "1024x1024"}),
        ("auto", {"aspect_ratio": "auto"}),
        ("16:9", {"aspect_ratio": "16:9"}),
        ("1K", {"size": "1K"}),
    ],
)
def test_size_fields(size: str, expected: dict[str, str]) -> None:
    fields = polza_size_fields(size)
    assert fields == expected
    assert "image_resolution" not in fields


def test_image_resolution_tiers_documented_only() -> None:
    assert image_resolution_tier(1536, 1024) == "1K"
    assert image_resolution_tier(2048, 1024) == "2K"
    assert image_resolution_tier(3840, 2160) == "4K"


def test_parse_completed_fixtures(
    polza_fixture: Callable[[str], dict[str, Any]],
) -> None:
    generated = parse_media_payload(polza_fixture("generate_completed"))
    assert generated.images and generated.images[0]["type"] == "url"
    assert generated.cost_rub == pytest.approx(2.37166294)
    edited = parse_media_payload(polza_fixture("edit_completed"))
    assert edited.cost_rub == pytest.approx(2.46868821)
    normalized = edited.as_images_response()
    assert normalized["data"][0]["url"].startswith("https://s3.polza.ai/")
    assert normalized["usage"]["cost_rub"] == pytest.approx(2.46868821)


def test_parse_accepts_object_data_and_b64() -> None:
    result = parse_media_payload(
        {"status": "completed", "data": {"b64_json": "QUJD"}, "usage": {"cost": 1.5}}
    )
    assert result.images == [{"type": "base64", "value": "QUJD"}]
    assert result.cost_rub == 1.5
    assert result.as_images_response()["usage"]["cost_rub"] == 1.5


def test_error_fixture_maps_without_internals(
    polza_fixture: Callable[[str], dict[str, Any]],
) -> None:
    body = polza_fixture("error_bad_request")
    with structlog.testing.capture_logs() as logs:
        exc = map_media_error(body, status_code=400)
    assert exc.error_code == "BAD_REQUEST"
    assert exc.status_code == 400
    trace_id = body["trace_id"]
    assert trace_id not in exc.message
    assert "input.prompt" not in exc.message
    logged = [entry for entry in logs if entry["event"] == "polza_media_error"]
    assert logged and logged[0]["trace_id"] == trace_id


def test_error_logs_reason_and_provider_name() -> None:
    body = {
        "error": {
            "code": "BAD_GATEWAY",
            "message": "upstream down",
            "metadata": {"reason": "upstream_error", "provider_name": "OpenAI"},
        },
        "trace_id": "t-1",
    }
    with structlog.testing.capture_logs() as logs:
        exc = map_media_error(body, status_code=502)
    assert exc.error_code == "BAD_GATEWAY"
    entry = next(item for item in logs if item["event"] == "polza_media_error")
    assert entry["reason"] == "upstream_error"
    assert entry["provider_name"] == "OpenAI"


@pytest.mark.asyncio
async def test_generate_sync_completed_sends_expected_request(
    polza_fixture: Callable[[str], dict[str, Any]],
) -> None:
    provider = make_polza_provider()
    with respx.mock(assert_all_called=True) as router:
        route = router.post(_MEDIA_URL).mock(
            return_value=httpx.Response(200, json=polza_fixture("generate_completed"))
        )
        async with httpx.AsyncClient() as client:
            result = await generate(
                client,
                provider,
                model=_MODEL,
                prompt="постер",
                aspect_ratio="3:2",
                size="1536x1024",
                quality="high",
            )
    request = route.calls[0].request
    sent = json.loads(request.content)
    assert sent["model"] == _MODEL
    assert "async" not in sent
    assert result.images[0]["value"].startswith("https://s3.polza.ai/")
    assert result.cost_rub == pytest.approx(2.37166294)


@pytest.mark.asyncio
async def test_generate_logs_warnings(
    polza_fixture: Callable[[str], dict[str, Any]],
) -> None:
    payload = polza_fixture("generate_completed")
    payload["warnings"] = ["unsupported parameter ignored: complexity"]
    with respx.mock() as router, structlog.testing.capture_logs() as logs:
        router.post(_MEDIA_URL).mock(return_value=httpx.Response(200, json=payload))
        async with httpx.AsyncClient() as client:
            await generate(client, make_polza_provider(), model=_MODEL, prompt="x")
    warned = [entry for entry in logs if entry["event"] == "polza_media_warnings"]
    assert warned
    assert "complexity" in str(warned[0]["warnings"])


@pytest.mark.asyncio
async def test_generate_http_400_maps_to_assist_error(
    polza_fixture: Callable[[str], dict[str, Any]],
) -> None:
    with respx.mock() as router:
        router.post(_MEDIA_URL).mock(
            return_value=httpx.Response(400, json=polza_fixture("error_bad_request"))
        )
        async with httpx.AsyncClient() as client:
            with pytest.raises(AssistAIError) as info:
                await generate(client, make_polza_provider(), model=_MODEL, prompt="x")
    assert info.value.error_code == "BAD_REQUEST"
    assert info.value.status_code == 400


@pytest.mark.asyncio
async def test_generate_rejects_more_than_limit() -> None:
    async with httpx.AsyncClient() as client:
        with pytest.raises(AssistAIError):
            await generate(
                client,
                make_polza_provider(),
                model=_MODEL,
                prompt="x",
                images=[b"x"] * (POLZA_MAX_REFERENCE_IMAGES + 1),
            )


@pytest.mark.asyncio
async def test_client_edit_two_photos_sends_two_references(
    multi_client_factory: Callable[..., AssistAIClient],
    polza_fixture: Callable[[str], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ALLOW_COMPOSITE_FALLBACK", raising=False)
    client = multi_client_factory()
    base, extra = _png((255, 0, 0), (64, 48)), _png((0, 0, 255), (64, 48))
    set_billing_user(4242)
    try:
        with respx.mock() as router:
            route = router.post(_MEDIA_URL).mock(
                return_value=httpx.Response(200, json=polza_fixture("edit_completed"))
            )
            images = await client.edit_images(
                [(base, "a.png", "image/png"), (extra, "b.png", "image/png")],
                "объедини",
                labels=["1", "2"],
            )
    finally:
        set_billing_user(None)
    assert images[0].startswith("https://s3.polza.ai/")
    sent = json.loads(route.calls[0].request.content)
    refs = sent["input"]["images"]
    assert len(refs) == 2
    assert base64.b64decode(refs[0]["data"]) == base
    assert base64.b64decode(refs[1]["data"]) == extra
    assert "composite" not in sent["input"]["prompt"]
    assert "complexity" not in sent["input"]
    assert sent["provider"]["only"] == ["OpenAI"]
    assert sent["user"] == hash_user_id(4242)
    assert "4242" not in sent["user"]


@pytest.mark.asyncio
async def test_client_generate_sends_size_not_resolution(
    multi_client_factory: Callable[..., AssistAIClient],
    polza_fixture: Callable[[str], dict[str, Any]],
) -> None:
    client = multi_client_factory()
    with respx.mock() as router:
        route = router.post(_MEDIA_URL).mock(
            return_value=httpx.Response(200, json=polza_fixture("generate_completed"))
        )
        await client.generate_image("постер", model=_MODEL, size="1536x1024")
    sent = json.loads(route.calls[0].request.content)
    media_input = sent["input"]
    assert media_input["aspect_ratio"] == "3:2"
    assert media_input["size"] == "1536x1024"
    assert media_input["quality"] == "high"
    assert "image_resolution" not in media_input
    assert "complexity" not in media_input
    assert "background" not in media_input


# --- G7: capability-aware multi-image ---


def test_multi_image_depends_on_provider_and_model() -> None:
    polza = make_polza_provider()
    assert supports_multi_image(_MODEL)
    assert supports_multi_image(_MODEL, polza)
    assert not supports_multi_image("black-forest-labs/flux-3-image", polza)
    assert max_edit_images(polza) == POLZA_MAX_REFERENCE_IMAGES == 16
    assert max_edit_images(None) is None


def test_polza_album_is_not_composited(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALLOW_COMPOSITE_FALLBACK", raising=False)
    files = [(_png((i, i, i)), f"{i}.png", "image/png") for i in range(5)]
    prepared = prepare_edit_files(files, _MODEL, provider=make_polza_provider())
    assert len(prepared) == 5
    assert prepared[0][1] == "0.png"


def test_polza_limit_16_enforced() -> None:
    files = [(b"x", f"{i}.png", "image/png") for i in range(17)]
    polza: ProviderConfig = make_polza_provider()
    with pytest.raises(AssistAIError):
        prepare_edit_files(files, _MODEL, provider=polza)
    # ProxyAPI (provider=None) — прежнее поведение, лимита нет.
    assert len(prepare_edit_files(files, _MODEL)) == 17
