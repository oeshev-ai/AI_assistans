"""Тесты Yandex SpeechKit TTS v3."""

from __future__ import annotations

import base64
import json

import httpx
import pytest
from respx import MockRouter

from Assist_ai import AssistAIClient, AssistAIConfig, AssistAIError, SpeechResult
from speechkit_tts import (
    SpeechKitConfig,
    SpeechKitError,
    is_speechkit_tts_model,
    parse_speechkit_audio,
    resolve_speechkit_role,
    resolve_speechkit_voice,
    speechkit_roles_for,
    synthesize,
)

SPEECHKIT_URL = "https://tts.api.cloud.yandex.net/tts/v3/utteranceSynthesis"


def test_resolve_speechkit_voice_legacy() -> None:
    assert resolve_speechkit_voice("alloy") == "marina"
    assert resolve_speechkit_voice("alexander") == "alexander"
    assert resolve_speechkit_voice(None, default="dasha") == "dasha"


def test_is_speechkit_tts_model() -> None:
    assert is_speechkit_tts_model("yandex/speechkit")
    assert is_speechkit_tts_model("Yandex/SpeechKit")
    assert not is_speechkit_tts_model("openai/gpt-4o-mini-tts")
    assert not is_speechkit_tts_model(None)


def test_speechkit_roles_per_voice() -> None:
    assert speechkit_roles_for("filipp") == ()
    assert "friendly" in speechkit_roles_for("marina")
    assert "friendly" not in speechkit_roles_for("alexander")
    assert resolve_speechkit_role("filipp", "friendly") == ""
    assert resolve_speechkit_role("alexander", "friendly") == ""
    assert resolve_speechkit_role("alexander", "good") == "good"
    assert resolve_speechkit_role("marina", "FRIENDLY") == "friendly"


def test_parse_speechkit_audio_ndjson() -> None:
    part1 = base64.b64encode(b"AAA").decode()
    part2 = base64.b64encode(b"BBB").decode()
    body = "\n".join(
        [
            json.dumps({"result": {"audioChunk": {"data": part1}}}),
            json.dumps({"result": {"audioChunk": {"data": part2}}}),
        ]
    ).encode()
    assert parse_speechkit_audio(body) == b"AAABBB"


def test_parse_speechkit_audio_single_json() -> None:
    data = base64.b64encode(b"ogg").decode()
    body = json.dumps({"result": {"audioChunk": {"data": data}}}).encode()
    assert parse_speechkit_audio(body) == b"ogg"


@pytest.mark.asyncio
async def test_synthesize_success(respx_mock: MockRouter) -> None:
    chunk = base64.b64encode(b"ogg-bytes").decode()
    respx_mock.post(SPEECHKIT_URL).mock(
        return_value=httpx.Response(
            200,
            content=json.dumps({"result": {"audioChunk": {"data": chunk}}}).encode(),
        )
    )
    config = SpeechKitConfig(
        api_key="test-key",
        folder_id="folder",
        voice="marina",
        role="friendly",
    )
    async with httpx.AsyncClient() as http:
        audio = await synthesize("Привет", config=config, http=http)
    assert audio == b"ogg-bytes"


@pytest.mark.asyncio
async def test_synthesize_http_403(respx_mock: MockRouter) -> None:
    respx_mock.post(SPEECHKIT_URL).mock(
        return_value=httpx.Response(403, text="forbidden")
    )
    config = SpeechKitConfig(api_key="bad", folder_id="folder")
    async with httpx.AsyncClient() as http:
        with pytest.raises(SpeechKitError, match="HTTP 403") as captured:
            await synthesize("Привет", config=config, http=http)
    assert captured.value.status_code == 403


@pytest.mark.asyncio
async def test_synthesize_empty_audio(respx_mock: MockRouter) -> None:
    respx_mock.post(SPEECHKIT_URL).mock(
        return_value=httpx.Response(200, content=b'{"result":{}}')
    )
    config = SpeechKitConfig(api_key="test-key", folder_id="folder")
    async with httpx.AsyncClient() as http:
        with pytest.raises(SpeechKitError, match="пустой аудиоответ"):
            await synthesize("Привет", config=config, http=http)


@pytest.mark.asyncio
async def test_assist_client_speechkit_route(respx_mock: MockRouter) -> None:
    chunk = base64.b64encode(b"voice-ogg").decode()
    respx_mock.post(SPEECHKIT_URL).mock(
        return_value=httpx.Response(
            200,
            content=json.dumps({"result": {"audioChunk": {"data": chunk}}}).encode(),
        )
    )
    client = AssistAIClient(
        AssistAIConfig(
            proxyapi_key="test-proxy-key",
            tts_provider="speechkit",
            speechkit_api_key="sk-key",
            speechkit_folder_id="b1folder",
            default_tts_model="yandex/speechkit",
            default_tts_voice="marina",
            speechkit_role="friendly",
            timeout=5.0,
        )
    )
    try:
        speech = await client.synthesize_speech("Привет", voice="alexander")
        assert isinstance(speech, SpeechResult)
        assert speech.audio == b"voice-ogg"
        assert speech.filename == "answer.ogg"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_assist_client_speechkit_missing_key() -> None:
    client = AssistAIClient(
        AssistAIConfig(
            proxyapi_key="test-proxy-key",
            tts_provider="speechkit",
            speechkit_api_key="",
            default_tts_model="yandex/speechkit",
            timeout=5.0,
        )
    )
    try:
        with pytest.raises(AssistAIError, match="YC_SPEECHKIT_API_KEY"):
            await client.synthesize_speech("Привет")
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_assist_client_routes_by_model_not_provider(
    respx_mock: MockRouter,
) -> None:
    """Даже при TTS_PROVIDER=proxyapi модель yandex/speechkit идёт в SpeechKit."""
    chunk = base64.b64encode(b"sk-ogg").decode()
    respx_mock.post(SPEECHKIT_URL).mock(
        return_value=httpx.Response(
            200,
            content=json.dumps({"result": {"audioChunk": {"data": chunk}}}).encode(),
        )
    )
    client = AssistAIClient(
        AssistAIConfig(
            proxyapi_key="test-proxy-key",
            tts_provider="proxyapi",
            speechkit_api_key="sk-key",
            speechkit_folder_id="folder",
            default_tts_model="openai/gpt-4o-mini-tts",
            default_tts_voice="alloy",
            timeout=5.0,
        )
    )
    try:
        speech = await client.synthesize_speech(
            "Привет", model="yandex/speechkit", voice="marina"
        )
        assert speech.audio == b"sk-ogg"
    finally:
        await client.close()
