"""Фактический учёт chat/STT/TTS по Polza ``cost_rub`` и метки service."""

from __future__ import annotations

import base64
import json
from collections.abc import Callable, MutableMapping
from typing import Any

import httpx
import pytest
import respx
import structlog
from prometheus_client import REGISTRY

from app.infra.cost_tracker import (
    CostTracker,
    chat_cost_usd,
    extract_cost_rub,
)
from app.llm.assist_ai import AssistAIClient
from app.llm.providers import ProviderConfig
from app.media.media_tools import (
    MediaTurnScope,
    SynthesizeSpeechTool,
    TranscribeMediaTool,
    UserMedia,
    set_user_media,
)
from tests.polza_helpers import make_polza_provider

_POLZA_CHAT = "https://polza.ai/api/v1/chat/completions"
_PROXY_CHAT = "https://api.proxyapi.ru/v1/chat/completions"
_POLZA_STT = "https://polza.ai/api/v1/audio/transcriptions"
_PROXY_STT = "https://api.proxyapi.ru/v1/audio/transcriptions"
_POLZA_TTS = "https://polza.ai/api/v1/audio/speech"
_PROXY_TTS = "https://api.proxyapi.ru/v1/audio/speech"
_MODEL = "openai/gpt-4o"
_ROUTES_TEXT = {
    "text": "polza",
    "image": "proxyapi",
    "audio": "proxyapi",
    "video": "proxyapi",
}
_ROUTES_AUDIO = {
    "text": "proxyapi",
    "image": "proxyapi",
    "audio": "polza",
    "video": "proxyapi",
}


def _sample(kind: str, source: str, provider: str, model: str) -> float:
    value = REGISTRY.get_sample_value(
        "llm_cost_usd_total",
        {"kind": kind, "provider": provider, "model": model, "source": source},
    )
    return value or 0.0


def _calls(service: str, endpoint: str, status: str) -> float:
    value = REGISTRY.get_sample_value(
        "api_calls_total",
        {"service": service, "endpoint": endpoint, "status": status},
    )
    return value or 0.0


def _spend(
    logs: list[MutableMapping[str, Any]], kind: str
) -> list[MutableMapping[str, Any]]:
    return [
        item
        for item in logs
        if item.get("event") == "llm_spend_recorded" and item.get("kind") == kind
    ]


def _chat_body(content: str, usage: dict[str, Any] | None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": _MODEL,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
    }
    if usage is not None:
        body["usage"] = usage
    return body


def _sse(events: list[dict[str, Any]]) -> str:
    lines = ["data: " + json.dumps(event, ensure_ascii=False) for event in events]
    lines.append("data: [DONE]")
    return "\n".join(lines) + "\n"


def _proxy_provider() -> ProviderConfig:
    return ProviderConfig(
        id="proxyapi",
        kind="openai",
        base_url="https://api.proxyapi.ru/v1",
        api_key="test-proxy-key",
        capabilities=frozenset({"chat", "audio"}),
        cost_source="usage_tokens",
        cost_multiplier=1.0,
    )


def test_extract_cost_rub_from_field() -> None:
    assert extract_cost_rub({"cost_rub": 1.52}) == 1.52
    assert extract_cost_rub({"cost_rub": 0}) == 0.0


def test_extract_cost_rub_alias_cost() -> None:
    assert extract_cost_rub({"cost": 2.5}) == 2.5
    estimate = CostTracker.estimate_image_cost_detailed(
        1,
        model="openai/gpt-image-2.6-flare",
        usage={"cost": 304.0, "output_tokens": 10},
        rub_per_usd=304.0,
    )
    assert estimate.source == "cost_rub"
    assert estimate.amount_usd == pytest.approx(1.0)
    assert estimate.cost_rub == 304.0


def test_extract_cost_rub_missing_usage() -> None:
    assert extract_cost_rub(None) is None
    assert extract_cost_rub({}) is None
    assert extract_cost_rub({"prompt_tokens": 3}) is None
    assert extract_cost_rub({"cost_rub": "1.52", "cost": 4}) == 4


def test_extract_cost_rub_rejects_negative_nan_and_bool() -> None:
    assert extract_cost_rub({"cost_rub": -0.1, "cost": 3}) is None
    assert extract_cost_rub({"cost_rub": float("nan")}) is None
    assert extract_cost_rub({"cost_rub": float("inf")}) is None
    assert extract_cost_rub({"cost_rub": True, "cost": 1.25}) == 1.25
    assert extract_cost_rub({"cost": False}) is None


def test_chat_cost_usd_prefers_rub_then_tokens_then_chars() -> None:
    polza = make_polza_provider(cost_multiplier=0.39)
    amount, source, rub = chat_cost_usd(
        usage={"cost_rub": 1.52, "prompt_tokens": 1000, "completion_tokens": 1000},
        provider=polza,
        rub_per_usd=304.0,
        prompt_tokens=1000,
        completion_tokens=1000,
        input_chars=400,
        output_chars=80,
    )
    assert source == "cost_rub"
    assert rub == 1.52
    assert amount == pytest.approx(1.52 / 304.0)

    plain = CostTracker.estimate_chat_cost(prompt_tokens=1000, completion_tokens=1000)
    scaled, source, rub = chat_cost_usd(
        usage={"prompt_tokens": 1000, "completion_tokens": 1000},
        provider=polza,
        rub_per_usd=304.0,
        prompt_tokens=1000,
        completion_tokens=1000,
    )
    assert source == "usage"
    assert rub is None
    assert scaled == pytest.approx(plain * 0.39)

    proxy_amount, source, rub = chat_cost_usd(
        usage={"prompt_tokens": 1000, "completion_tokens": 1000},
        provider=_proxy_provider(),
        rub_per_usd=304.0,
        prompt_tokens=1000,
        completion_tokens=1000,
    )
    assert source == "usage"
    assert rub is None
    assert proxy_amount == pytest.approx(plain)

    flat, source, rub = chat_cost_usd(
        usage=None,
        provider=polza,
        rub_per_usd=304.0,
        input_chars=8,
        output_chars=4,
    )
    heuristic = CostTracker.estimate_chat_cost(input_chars=8, output_chars=4)
    assert source == "flat"
    assert rub is None
    assert flat == pytest.approx(heuristic * 0.39)


@pytest.mark.asyncio
async def test_record_chat_usage_cost_rub(
    multi_client_factory: Callable[..., AssistAIClient],
    assist_client: AssistAIClient,
) -> None:
    polza = multi_client_factory(routes=_ROUTES_TEXT, rub_per_usd=304.0)
    usage = {
        "prompt_tokens": 11,
        "completion_tokens": 2,
        "cost_rub": 1.52,
        "cost": 1.52,
    }
    before = _sample("chat", "cost_rub", "polza", _MODEL)
    with respx.mock(assert_all_called=False) as router:
        router.post(_POLZA_CHAT).mock(
            return_value=httpx.Response(200, json=_chat_body("ok", usage))
        )
        with structlog.testing.capture_logs() as logs:
            reply = await polza.chat_completion(
                [{"role": "user", "content": "hi"}],
                use_tools=False,
            )
    assert reply == "ok"
    assert _sample("chat", "cost_rub", "polza", _MODEL) == pytest.approx(
        before + 1.52 / 304.0
    )
    event = _spend(logs, "chat")[-1]
    assert event["source"] == "cost_rub"
    assert event["cost_rub"] == 1.52
    assert event["amount_usd"] == pytest.approx(0.005)
    assert event["provider"] == "polza"

    token_usage = {"prompt_tokens": 100, "completion_tokens": 20}
    expected = CostTracker.estimate_chat_cost(prompt_tokens=100, completion_tokens=20)
    before_usage = _sample("chat", "usage", "proxyapi", _MODEL)
    with respx.mock(assert_all_called=False) as router:
        router.post(_PROXY_CHAT).mock(
            return_value=httpx.Response(200, json=_chat_body("ok", token_usage))
        )
        with structlog.testing.capture_logs() as logs:
            reply = await assist_client.chat_completion(
                [{"role": "user", "content": "hi"}],
                use_tools=False,
            )
    assert reply == "ok"
    assert _sample("chat", "usage", "proxyapi", _MODEL) == pytest.approx(
        before_usage + expected
    )
    event = _spend(logs, "chat")[-1]
    assert event["source"] == "usage"
    assert event["cost_rub"] is None


@pytest.mark.asyncio
async def test_stream_chat_bills_final_usage_chunk(
    multi_client_factory: Callable[..., AssistAIClient],
) -> None:
    client = multi_client_factory(routes=_ROUTES_TEXT, rub_per_usd=304.0)
    stream = _sse(
        [
            {
                "choices": [
                    {"index": 0, "delta": {"content": "ok"}, "finish_reason": None}
                ]
            },
            {
                "choices": [],
                "model": _MODEL,
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 1,
                    "cost_rub": 1.52,
                    "cost": 1.52,
                },
            },
        ]
    )
    before = _sample("chat", "cost_rub", "polza", _MODEL)
    with respx.mock(assert_all_called=False) as router:
        router.post(_POLZA_CHAT).mock(return_value=httpx.Response(200, text=stream))
        with structlog.testing.capture_logs() as logs:
            chunks = [
                delta
                async for delta in client.stream_chat(
                    [{"role": "user", "content": "hi"}],
                    use_tools=False,
                )
            ]
    assert "".join(delta.text for delta in chunks) == "ok"
    spends = _spend(logs, "chat")
    assert len(spends) == 1
    assert spends[0]["source"] == "cost_rub"
    assert spends[0]["cost_rub"] == 1.52
    assert spends[0]["provider"] == "polza"
    assert _sample("chat", "cost_rub", "polza", _MODEL) == pytest.approx(
        before + 1.52 / 304.0
    )


@pytest.mark.asyncio
async def test_stream_chat_falls_back_without_usage(
    multi_client_factory: Callable[..., AssistAIClient],
) -> None:
    client = multi_client_factory(routes=_ROUTES_TEXT, rub_per_usd=304.0)
    stream = _sse(
        [
            {
                "choices": [
                    {"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}
                ]
            }
        ]
    )
    with respx.mock(assert_all_called=False) as router:
        router.post(_POLZA_CHAT).mock(return_value=httpx.Response(200, text=stream))
        with structlog.testing.capture_logs() as logs:
            chunks = [
                delta
                async for delta in client.stream_chat(
                    [{"role": "user", "content": "hi"}],
                    use_tools=False,
                )
            ]
    assert "".join(delta.text for delta in chunks) == "ok"
    spends = _spend(logs, "chat")
    assert len(spends) == 1
    assert spends[0]["source"] == "flat"
    assert spends[0]["cost_rub"] is None
    heuristic = CostTracker.estimate_chat_cost(input_chars=2, output_chars=2)
    assert spends[0]["amount_usd"] == pytest.approx(round(heuristic * 0.39, 6))


@pytest.mark.asyncio
async def test_stt_record_cost_rub(
    multi_client_factory: Callable[..., AssistAIClient],
    assist_client: AssistAIClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_prepare(data: bytes, *, filename: str, is_video: bool) -> list:
        return [(data, filename)]

    monkeypatch.setattr("app.media.media_tools.prepare_audio_for_stt", fake_prepare)
    polza = multi_client_factory(routes=_ROUTES_AUDIO, rub_per_usd=304.0)
    tool = TranscribeMediaTool(polza)
    token = set_user_media(
        [
            UserMedia(
                kind="audio",
                filename="voice.ogg",
                mime_type="audio/ogg",
                data=b"ogg-bytes",
            )
        ]
    )
    try:
        with respx.mock(assert_all_called=False) as router:
            router.post(_POLZA_STT).mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "text": "привет",
                        "usage": {"cost_rub": 0.11, "cost": 0.11, "durationSeconds": 3},
                    },
                )
            )
            with structlog.testing.capture_logs() as logs:
                raw = await tool.execute()
        payload = json.loads(raw)
        assert payload["status"] == "success"
        event = _spend(logs, "stt")[-1]
        assert event["source"] == "cost_rub"
        assert event["cost_rub"] == 0.11
        assert event["amount_usd"] == pytest.approx(round(0.11 / 304.0, 6))
        assert event["provider"] == "polza"
    finally:
        from app.media.media_tools import reset_user_media

        reset_user_media(token)

    proxy_tool = TranscribeMediaTool(assist_client)
    audio = b"x" * 4000
    estimated = CostTracker.estimate_stt_cost(max(15.0, len(audio) / 4000.0))
    token = set_user_media(
        [
            UserMedia(
                kind="audio",
                filename="voice.ogg",
                mime_type="audio/ogg",
                data=audio,
            )
        ]
    )
    try:
        with respx.mock(assert_all_called=False) as router:
            router.post(_PROXY_STT).mock(
                return_value=httpx.Response(200, json={"text": "без цены"})
            )
            with structlog.testing.capture_logs() as logs:
                raw = await proxy_tool.execute()
        assert json.loads(raw)["status"] == "success"
        event = _spend(logs, "stt")[-1]
        assert event["source"] is None
        assert event["cost_rub"] is None
        assert event["amount_usd"] == pytest.approx(round(estimated, 6))
    finally:
        from app.media.media_tools import reset_user_media

        reset_user_media(token)


@pytest.mark.asyncio
async def test_tts_record_cost_rub(
    multi_client_factory: Callable[..., AssistAIClient],
    assist_client: AssistAIClient,
) -> None:
    polza = multi_client_factory(routes=_ROUTES_AUDIO, rub_per_usd=304.0)
    tool = SynthesizeSpeechTool(polza)
    encoded = base64.b64encode(b"ogg-bytes").decode("ascii")
    with (
        MediaTurnScope(),
        respx.mock(assert_all_called=False) as router,
    ):
        router.post(_POLZA_TTS).mock(
            return_value=httpx.Response(
                200,
                json={
                    "audio": encoded,
                    "contentType": "audio/ogg",
                    "usage": {"characters": 6, "cost_rub": 0.11, "cost": 0.11},
                },
            )
        )
        with structlog.testing.capture_logs() as logs:
            raw = await tool.execute(text="Привет")
    assert json.loads(raw)["status"] == "success"
    event = _spend(logs, "tts")[-1]
    assert event["source"] == "cost_rub"
    assert event["cost_rub"] == 0.11
    assert event["amount_usd"] == pytest.approx(round(0.11 / 304.0, 6))
    assert event["provider"] == "polza"

    proxy_tool = SynthesizeSpeechTool(assist_client)
    estimated = CostTracker.estimate_tts_cost("Привет")
    with respx.mock(assert_all_called=False) as router:
        router.post(_PROXY_TTS).mock(
            return_value=httpx.Response(200, content=b"proxy-ogg")
        )
        with structlog.testing.capture_logs() as logs:
            raw = await proxy_tool.execute(text="Привет")
    assert json.loads(raw)["status"] == "success"
    event = _spend(logs, "tts")[-1]
    assert event["source"] is None
    assert event["cost_rub"] is None
    assert event["amount_usd"] == pytest.approx(round(estimated, 6))


@pytest.mark.asyncio
async def test_transport_metrics_service_label(
    multi_client_factory: Callable[..., AssistAIClient],
) -> None:
    client = multi_client_factory(routes=_ROUTES_TEXT, rub_per_usd=304.0)
    body = _chat_body("ok", {"prompt_tokens": 1, "completion_tokens": 1, "cost_rub": 1})
    before_polza = _calls("polza", "post_json", "success")
    with respx.mock(assert_all_called=False) as router:
        router.post(_POLZA_CHAT).mock(return_value=httpx.Response(200, json=body))
        await client.chat_completion(
            [{"role": "user", "content": "hi"}],
            use_tools=False,
        )
    assert _calls("polza", "post_json", "success") == before_polza + 1

    before_err = _calls("polza", "post_json", "error")
    before_proxy = _calls("proxyapi", "post_json", "success")
    with respx.mock(assert_all_called=False) as router:
        router.post(_POLZA_CHAT).mock(
            return_value=httpx.Response(503, json={"error": {"message": "down"}})
        )
        router.post(_PROXY_CHAT).mock(return_value=httpx.Response(200, json=body))
        reply = await client.chat_completion(
            [{"role": "user", "content": "hi"}],
            use_tools=False,
        )
    assert reply == "ok"
    assert _calls("polza", "post_json", "error") == before_err + 1
    assert _calls("proxyapi", "post_json", "success") == before_proxy + 1

    before_stream = _calls("polza", "chat_stream", "success")
    stream = _sse(
        [
            {
                "choices": [
                    {"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}
                ]
            }
        ]
    )
    with respx.mock(assert_all_called=False) as router:
        router.post(_POLZA_CHAT).mock(return_value=httpx.Response(200, text=stream))
        _ = [
            delta
            async for delta in client.stream_chat(
                [{"role": "user", "content": "hi"}],
                use_tools=False,
            )
        ]
    assert _calls("polza", "chat_stream", "success") == before_stream + 1
