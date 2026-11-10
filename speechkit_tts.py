"""Yandex SpeechKit TTS v3 (REST) — синтез речи в OGG Opus."""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from typing import Any

import httpx
import structlog

logger = structlog.get_logger("speechkit_tts")

DEFAULT_SPEECHKIT_URL = (
    "https://tts.api.cloud.yandex.net/tts/v3/utteranceSynthesis"
)
DEFAULT_VOICE = "marina"
DEFAULT_ROLE = "friendly"
DEFAULT_SPEED = 1.0

PROXYAPI_LEGACY_VOICES = frozenset({"alloy", "verse", "nova", "shimmer", "echo", "fable", "onyx"})

SPEECHKIT_VOICES: tuple[str, ...] = (
    "marina",
    "alexander",
    "alena",
    "dasha",
    "julia",
    "lera",
    "masha",
    "kirill",
    "anton",
    "zahar",
    "jane",
    "filipp",
)


class SpeechKitError(Exception):
    """Ошибка синтеза SpeechKit."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class SpeechKitConfig:
    api_key: str
    folder_id: str
    voice: str = DEFAULT_VOICE
    role: str = DEFAULT_ROLE
    speed: float = DEFAULT_SPEED
    url: str = DEFAULT_SPEECHKIT_URL

    @classmethod
    def from_env(cls) -> SpeechKitConfig | None:
        """Читает конфиг из env. ``None``, если ключ не задан."""
        api_key = os.getenv("YC_SPEECHKIT_API_KEY", "").strip()
        if not api_key:
            return None
        folder_id = os.getenv("YC_FOLDER_ID", "").strip()
        role = os.getenv("SPEECHKIT_ROLE", DEFAULT_ROLE).strip()
        speed_raw = os.getenv("SPEECHKIT_SPEED", str(DEFAULT_SPEED)).strip() or str(
            DEFAULT_SPEED
        )
        try:
            speed = float(speed_raw)
        except ValueError:
            speed = DEFAULT_SPEED
        voice = os.getenv("SPEECHKIT_VOICE", DEFAULT_VOICE).strip() or DEFAULT_VOICE
        url = (
            os.getenv("SPEECHKIT_URL", DEFAULT_SPEECHKIT_URL).strip()
            or DEFAULT_SPEECHKIT_URL
        )
        return cls(
            api_key=api_key,
            folder_id=folder_id,
            voice=voice,
            role=role,
            speed=speed,
            url=url,
        )


def resolve_speechkit_voice(voice: str | None, default: str = DEFAULT_VOICE) -> str:
    """Подставляет дефолт, если в профиле остался голос ProxyAPI."""
    cleaned = (voice or "").strip()
    if not cleaned or cleaned.casefold() in PROXYAPI_LEGACY_VOICES:
        return default
    return cleaned


def _extract_audio_chunks(payload: Any, out: list[bytes]) -> None:
    """Рекурсивно достаёт base64 ``audioChunk.data`` из JSON SpeechKit."""
    if isinstance(payload, list):
        for item in payload:
            _extract_audio_chunks(item, out)
        return
    if not isinstance(payload, dict):
        return
    chunk = payload.get("audioChunk")
    if isinstance(chunk, dict):
        data = chunk.get("data")
        if isinstance(data, str) and data:
            out.append(base64.b64decode(data))
    elif isinstance(chunk, list):
        for item in chunk:
            _extract_audio_chunks({"audioChunk": item}, out)
    result = payload.get("result")
    if isinstance(result, (dict, list)):
        _extract_audio_chunks(result, out)
    for value in payload.values():
        if value is chunk or value is result:
            continue
        if isinstance(value, (dict, list)):
            _extract_audio_chunks(value, out)


def parse_speechkit_audio(body: bytes) -> bytes:
    """Собирает OGG/WAV из тела ответа REST (NDJSON или один JSON)."""
    text = body.decode("utf-8", errors="replace").strip()
    if not text:
        return b""
    chunks: list[bytes] = []
    # Один JSON-документ
    try:
        parsed = json.loads(text)
        _extract_audio_chunks(parsed, chunks)
        if chunks:
            return b"".join(chunks)
    except json.JSONDecodeError:
        pass
    # NDJSON / несколько объектов подряд
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        _extract_audio_chunks(parsed, chunks)
    return b"".join(chunks)


def _build_request_body(
    text: str,
    *,
    voice: str,
    role: str,
    speed: float,
) -> dict[str, Any]:
    hints: list[dict[str, Any]] = [{"voice": voice}]
    if role:
        hints.append({"role": role})
    if speed and speed != 1.0:
        hints.append({"speed": speed})
    return {
        "text": text,
        "hints": hints,
        "outputAudioSpec": {
            "containerAudio": {"containerAudioType": "OGG_OPUS"},
        },
        "unsafeMode": True,
    }


async def synthesize(
    text: str,
    *,
    config: SpeechKitConfig,
    voice: str | None = None,
    role: str | None = None,
    speed: float | None = None,
    http: httpx.AsyncClient,
    timeout: float = 120.0,
) -> bytes:
    """Синтезирует речь через SpeechKit TTS v3 REST.

    Returns:
        Байты OGG Opus.

    Raises:
        SpeechKitError: пустой текст, HTTP/сеть или пустой аудиоответ.
        ValueError: пустой текст.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text не может быть пустым.")
    if not config.api_key:
        raise SpeechKitError("YC_SPEECHKIT_API_KEY не задан.")

    selected_voice = resolve_speechkit_voice(voice, config.voice)
    selected_role = (role if role is not None else config.role).strip()
    selected_speed = config.speed if speed is None else speed
    payload = _build_request_body(
        text.strip(),
        voice=selected_voice,
        role=selected_role,
        speed=selected_speed,
    )
    headers = {
        "Authorization": f"Api-Key {config.api_key}",
        "Content-Type": "application/json",
    }
    if config.folder_id:
        headers["x-folder-id"] = config.folder_id

    logger.info(
        "speechkit_synthesize",
        voice=selected_voice,
        role=selected_role or None,
        speed=selected_speed,
        chars=len(text),
    )
    try:
        response = await http.post(
            config.url,
            headers=headers,
            json=payload,
            timeout=httpx.Timeout(timeout, connect=min(30.0, timeout)),
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = (exc.response.text or "")[:500]
        logger.error(
            "speechkit_http_error",
            status=exc.response.status_code,
            detail=detail,
        )
        raise SpeechKitError(
            f"SpeechKit вернул HTTP {exc.response.status_code}: {detail}",
            status_code=exc.response.status_code,
        ) from exc
    except httpx.TimeoutException as exc:
        logger.error("speechkit_timeout")
        raise SpeechKitError("Превышено время ожидания ответа SpeechKit.") from exc
    except httpx.RequestError as exc:
        logger.error("speechkit_network_error", error=str(exc))
        raise SpeechKitError(f"Сетевая ошибка SpeechKit: {exc}") from exc

    audio = parse_speechkit_audio(response.content)
    if not audio:
        raise SpeechKitError("SpeechKit вернул пустой аудиоответ.")
    logger.info("speechkit_synthesize_ok", bytes=len(audio))
    return audio
