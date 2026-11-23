"""STT и TTS AssistAIClient. HTTP только через respx."""

from __future__ import annotations

import base64
from collections.abc import Callable

import httpx
import pytest
import respx
import structlog
from prometheus_client import REGISTRY
from respx import MockRouter

from app.llm.assist_ai import AssistAIClient
from app.llm.errors import AssistAIError
from app.llm.media_api import SpeechResult

TRANSCRIBE_URL = "https://api.proxyapi.ru/v1/audio/transcriptions"
SPEECH_URL = "https://api.proxyapi.ru/v1/audio/speech"
_POLZA_SPEECH = "https://polza.ai/api/v1/audio/speech"


def _fallback_count(source: str, target: str, reason: str) -> float:
    value = REGISTRY.get_sample_value(
        "provider_fallback_total", {"from": source, "to": target, "reason": reason}
    )
    return value or 0.0


@pytest.mark.asyncio
async def test_transcribe_audio_success(
    assist_client: AssistAIClient,
    respx_mock: MockRouter,
) -> None:
    respx_mock.post(TRANSCRIBE_URL).mock(
        return_value=httpx.Response(200, json={"text": "привет"})
    )

    text = await assist_client.transcribe_audio(b"fake-ogg", filename="voice.ogg")

    assert text.text == "привет"
    assert text.cost_usd is None


@pytest.mark.asyncio
async def test_transcribe_audio_empty_text(
    assist_client: AssistAIClient,
    respx_mock: MockRouter,
) -> None:
    respx_mock.post(TRANSCRIBE_URL).mock(
        return_value=httpx.Response(200, json={"text": ""})
    )

    text = await assist_client.transcribe_audio(b"fake-ogg", filename="voice.ogg")

    assert text.text == ""
    assert text.cost_source is None


@pytest.mark.asyncio
async def test_transcribe_audio_timeout(
    assist_client: AssistAIClient,
    respx_mock: MockRouter,
) -> None:
    respx_mock.post(TRANSCRIBE_URL).mock(side_effect=httpx.ReadTimeout("timed out"))

    with pytest.raises(AssistAIError) as captured:
        await assist_client.transcribe_audio(b"fake-ogg", filename="voice.ogg")

    assert "Превышено время ожидания" in captured.value.message
    assert "audio/transcriptions" in captured.value.message


@pytest.mark.asyncio
async def test_synthesize_speech_success(
    assist_client: AssistAIClient,
    respx_mock: MockRouter,
) -> None:
    payload = b"ogg-bytes"
    respx_mock.post(SPEECH_URL).mock(return_value=httpx.Response(200, content=payload))

    speech = await assist_client.synthesize_speech("Привет")

    assert isinstance(speech, SpeechResult)
    assert speech.audio == payload
    assert speech.filename == "answer.ogg"
    assert speech.mime_type == "audio/ogg"


@pytest.mark.asyncio
async def test_synthesize_speech_empty_audio(
    assist_client: AssistAIClient,
    respx_mock: MockRouter,
) -> None:
    respx_mock.post(SPEECH_URL).mock(return_value=httpx.Response(200, content=b""))

    with pytest.raises(AssistAIError, match="пустой аудиоответ"):
        await assist_client.synthesize_speech("Привет")


@pytest.mark.asyncio
async def test_synthesize_speech_polza_json_base64(
    multi_client_factory: Callable[..., AssistAIClient],
) -> None:
    client = multi_client_factory(
        routes={"audio": "polza", "image": "polza"},
        audio_fallback="proxyapi",
    )
    payload = b"ogg-from-polza"
    with respx.mock(assert_all_called=False) as router:
        polza = router.post(_POLZA_SPEECH).mock(
            return_value=httpx.Response(
                200,
                json={
                    "audio": base64.b64encode(payload).decode("ascii"),
                    "contentType": "audio/ogg",
                },
            )
        )
        proxy = router.post(SPEECH_URL).mock(
            return_value=httpx.Response(200, content=b"proxy")
        )
        speech = await client.synthesize_speech("Привет", response_format="opus")
    assert speech.audio == payload
    assert speech.filename == "answer.ogg"
    assert speech.mime_type == "audio/ogg"
    assert len(polza.calls) == 1
    assert not proxy.called


@pytest.mark.asyncio
async def test_synthesize_speech_pins_polza_when_stt_route_is_proxyapi(
    multi_client_factory: Callable[..., AssistAIClient],
) -> None:
    client = multi_client_factory(
        routes={
            "text": "proxyapi",
            "image": "proxyapi",
            "audio": "proxyapi",
            "video": "proxyapi",
        },
        tts_provider="polza",
    )
    payload = b"ogg-from-polza"
    with respx.mock(assert_all_called=False) as router:
        polza = router.post(_POLZA_SPEECH).mock(
            return_value=httpx.Response(
                200,
                json={
                    "audio": base64.b64encode(payload).decode("ascii"),
                    "contentType": "audio/ogg",
                },
            )
        )
        proxy = router.post(SPEECH_URL).mock(
            return_value=httpx.Response(200, content=b"proxy")
        )
        stt = router.post(TRANSCRIBE_URL).mock(
            return_value=httpx.Response(200, json={"text": "распознано"})
        )
        speech = await client.synthesize_speech("Привет")
        heard = await client.transcribe_audio(b"fake-ogg", filename="voice.ogg")
    assert speech.audio == payload
    assert speech.provider_id == "polza"
    assert heard.text == "распознано"
    assert len(polza.calls) == 1
    assert len(stt.calls) == 1
    assert not proxy.called


@pytest.mark.asyncio
async def test_synthesize_speech_falls_back_to_proxyapi(
    multi_client_factory: Callable[..., AssistAIClient],
) -> None:
    client = multi_client_factory(
        routes={"audio": "polza", "image": "polza"},
        audio_fallback="proxyapi",
    )
    before = _fallback_count("polza", "proxyapi", "http_5xx")
    with respx.mock(assert_all_called=False) as router:
        polza = router.post(_POLZA_SPEECH).mock(return_value=httpx.Response(503))
        proxy = router.post(SPEECH_URL).mock(
            return_value=httpx.Response(200, content=b"proxy-ogg")
        )
        with structlog.testing.capture_logs() as logs:
            speech = await client.synthesize_speech("Привет")
    assert speech.audio == b"proxy-ogg"
    assert len(polza.calls) == 1
    assert len(proxy.calls) == 1
    assert _fallback_count("polza", "proxyapi", "http_5xx") == before + 1
    event = next(item for item in logs if item["event"] == "provider_fallback")
    assert (event["from"], event["to"], event["reason"]) == (
        "polza",
        "proxyapi",
        "http_5xx",
    )


@pytest.mark.asyncio
async def test_synthesize_speech_client_error_no_fallback(
    multi_client_factory: Callable[..., AssistAIClient],
) -> None:
    client = multi_client_factory(
        routes={"audio": "polza", "image": "polza"},
        audio_fallback="proxyapi",
    )
    with respx.mock(assert_all_called=False) as router:
        polza = router.post(_POLZA_SPEECH).mock(return_value=httpx.Response(400))
        proxy = router.post(SPEECH_URL).mock(
            return_value=httpx.Response(200, content=b"proxy")
        )
        with pytest.raises(AssistAIError) as info:
            await client.synthesize_speech("Привет")
    assert info.value.status_code == 400
    assert len(polza.calls) == 1
    assert not proxy.called
