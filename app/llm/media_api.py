"""Аудио/видео API: STT, TTS, video generation."""

from __future__ import annotations

import asyncio
import ipaddress
import socket
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse, urlunparse

import httpx
import structlog

from app.infra.cost_tracker import VIDEO_USD, cost_tracker
from app.infra.security import security
from app.llm.errors import AssistAIError
from app.llm.providers import ProviderConfig
from app.llm.transport import (
    _SPEECH_PATH,
    _SPEECH_READ_TIMEOUT,
    _TRANSCRIBE_PATH,
    _TRANSCRIBE_READ_TIMEOUT,
    _VIDEO_POLL_INTERVAL,
    _VIDEO_POLL_MAX_SECONDS,
    _VIDEO_READ_TIMEOUT,
    _VIDEOS_PATH,
    ProviderTransportMixin,
    _response_error_text,
    _ssl_context,
)

logger = structlog.get_logger("AssistAI.media")

_VIDEO_FAILED_USER = "Генерация видео завершилась ошибкой. Попробуйте позже."
_VIDEO_POLL_USER = "Не удалось опросить статус видео. Попробуйте позже."
_VIDEO_DOWNLOAD_USER = "Не удалось скачать видео. Попробуйте позже."


def video_user_error(kind: str) -> str:
    """Stable Telegram-safe copy for video failures (no provider payloads/URLs)."""
    if kind == "poll":
        return _VIDEO_POLL_USER
    if kind == "download":
        return _VIDEO_DOWNLOAD_USER
    return _VIDEO_FAILED_USER


@dataclass(frozen=True)
class SpeechResult:
    """Готовый аудиоответ TTS: байты, имя файла и MIME."""

    audio: bytes
    filename: str
    mime_type: str


@dataclass(frozen=True)
class VideoResult:
    """Готовый ролик: байты файла, имя и MIME."""

    data: bytes
    filename: str
    mime_type: str = "video/mp4"


def _audio_mime(filename: str) -> str:
    """MIME по расширению аудиофайла для multipart STT."""
    lowered = filename.rsplit(".", 1)[-1].casefold() if "." in filename else ""
    return {
        "ogg": "audio/ogg",
        "oga": "audio/ogg",
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "m4a": "audio/mp4",
        "webm": "audio/webm",
    }.get(lowered, "application/octet-stream")


_TTS_FORMATS: dict[str, tuple[str, str]] = {
    "opus": ("answer.ogg", "audio/ogg"),
    "ogg": ("answer.ogg", "audio/ogg"),
    "mp3": ("answer.mp3", "audio/mpeg"),
    "wav": ("answer.wav", "audio/wav"),
    "aac": ("answer.aac", "audio/aac"),
    "flac": ("answer.flac", "audio/flac"),
    "pcm": ("answer.pcm", "audio/pcm"),
}


def normalize_tts_format(response_format: str | None, *, default: str = "opus") -> str:
    """Return a known TTS format or ``default``."""
    raw = (response_format or default).strip().casefold() or default
    if raw in _TTS_FORMATS:
        return raw
    return default if default in _TTS_FORMATS else "opus"


def _speech_file(response_format: str) -> tuple[str, str]:
    """Имя файла и MIME для ответа TTS."""
    fmt = normalize_tts_format(response_format)
    return _TTS_FORMATS[fmt]


def _ip_is_non_public(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def _host_is_non_public(hostname: str) -> bool:
    """True for loopback/private/link-local IPs and obvious local names."""
    lowered = hostname.strip().casefold().rstrip(".")
    if not lowered:
        return True
    if lowered in {"localhost", "metadata.google.internal"} or lowered.endswith(
        ".local"
    ):
        return True
    try:
        ip = ipaddress.ip_address(lowered)
    except ValueError:
        return False
    return _ip_is_non_public(ip)


def _public_resolved_ips(hostname: str) -> tuple[str, ...]:
    """Public A/AAAA records only; empty if resolve fails or any IP is private.

    Rejecting mixed public+private answers closes a common DNS-rebinding pattern.
    """
    try:
        infos = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    except OSError:
        return ()
    ips: list[str] = []
    seen: set[str] = set()
    for info in infos:
        raw_ip = info[4][0]
        try:
            ip = ipaddress.ip_address(raw_ip)
        except ValueError:
            continue
        if _ip_is_non_public(ip):
            return ()
        text = str(ip)
        if text not in seen:
            seen.add(text)
            ips.append(text)
    return tuple(ips)


def resolve_video_resource_url(
    candidate: str | None,
    *,
    provider_base_url: str,
    fallback: str,
) -> tuple[str, bool, tuple[str, ...]]:
    """Choose a video poll/content URL, auth flag, and pinned public IPs.

    - Same host as ``provider_base_url`` **and HTTPS** → fetch with auth (no pin).
    - Other **public HTTPS** hosts (CDN) → fetch without auth, pinned to resolved
      public IPs so a later DNS rebind cannot steer the TCP connect private.
    - HTTP (including same-host), private/loopback/metadata, or unsafe DNS →
      ``fallback``.
    """
    if not isinstance(candidate, str) or not candidate.strip():
        return fallback, True, ()
    raw = candidate.strip()
    try:
        parsed = urlparse(raw)
        provider = urlparse(provider_base_url)
    except ValueError:
        return fallback, True, ()
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return fallback, True, ()
    host = parsed.hostname.casefold()
    provider_host = (provider.hostname or "").casefold()
    if host == provider_host:
        # Bearer must not travel over plaintext, even to the provider host.
        if parsed.scheme != "https":
            return fallback, True, ()
        return raw, True, ()
    if _host_is_non_public(host):
        return fallback, True, ()
    pinned = _public_resolved_ips(host)
    if not pinned:
        return fallback, True, ()
    if parsed.scheme != "https":
        return fallback, True, ()
    return raw, False, pinned


def _url_with_pinned_ip(url: str, ip: str) -> str:
    """Rewrite ``url`` netloc to ``ip`` (keep path/query); Host/SNI stay original."""
    parsed = urlparse(url)
    host_for_url = f"[{ip}]" if ":" in ip else ip
    if parsed.port:
        netloc = f"{host_for_url}:{parsed.port}"
    else:
        netloc = host_for_url
    return urlunparse(
        (
            parsed.scheme,
            netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            parsed.fragment,
        )
    )


class MediaApiMixin(ProviderTransportMixin):
    """Миксин video/STT/TTS для ``AssistAIClient``."""

    async def generate_video(
        self,
        prompt: str,
        *,
        model: str | None = None,
        duration: int | None = None,
        resolution: str | None = None,
    ) -> VideoResult:
        """Создаёт видео-задачу, ждёт completion и скачивает mp4."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt не может быть пустым.")
        if self._closed:
            raise AssistAIError("Клиент уже закрыт.")

        selected_model = model or self.config.default_video_model
        sanitized_prompt = security.sanitize_prompt(prompt.strip())
        payload: dict[str, Any] = {
            "model": selected_model,
            "prompt": sanitized_prompt,
        }
        if isinstance(duration, int) and duration > 0:
            payload["duration"] = duration
        if isinstance(resolution, str) and resolution.strip():
            payload["resolution"] = resolution.strip()

        await cost_tracker.check_budget(estimated=VIDEO_USD)
        try:
            create_timeout = httpx.Timeout(
                max(self.config.timeout, _VIDEO_READ_TIMEOUT),
                connect=self.config.timeout,
            )
            logger.info(
                "generate_video_create",
                model=selected_model,
                prompt_chars=len(sanitized_prompt),
            )
            video_provider = self._endpoint("video")
            created = await self._post_json(
                _VIDEOS_PATH, payload, timeout=create_timeout, provider=video_provider
            )
            job_id = created.get("id")
            if not isinstance(job_id, str) or not job_id:
                raise AssistAIError("ProxyAPI не вернул id видео-задачи.")
            status_fallback = (
                f"{video_provider.base_url.rstrip('/')}/{_VIDEOS_PATH}/{job_id}"
            )
            status_url, status_with_auth, status_pins = resolve_video_resource_url(
                created.get("polling_url")
                if isinstance(created.get("polling_url"), str)
                else None,
                provider_base_url=video_provider.base_url,
                fallback=status_fallback,
            )

            deadline = time.monotonic() + _VIDEO_POLL_MAX_SECONDS
            status_payload: dict[str, Any] = created
            while time.monotonic() < deadline:
                status = str(status_payload.get("status") or "").casefold()
                if status in {"completed", "succeeded", "success"}:
                    break
                if status in {"failed", "error", "cancelled"}:
                    logger.warning(
                        "generate_video_failed",
                        status=status,
                        job_id=job_id,
                        payload_keys=sorted(status_payload.keys()),
                    )
                    raise AssistAIError(video_user_error("failed"))
                await asyncio.sleep(_VIDEO_POLL_INTERVAL)
                try:
                    response = await self._video_get(
                        video_provider,
                        status_url,
                        with_auth=status_with_auth,
                        pinned_ips=status_pins,
                        timeout=httpx.Timeout(30.0, connect=self.config.timeout),
                    )
                    response.raise_for_status()
                    body = response.json()
                except httpx.HTTPError as exc:
                    logger.warning("generate_video_poll_failed", error=str(exc))
                    raise AssistAIError(video_user_error("poll")) from exc
                if not isinstance(body, dict):
                    raise AssistAIError("Некорректный статус видео-задачи.")
                status_payload = body
            else:
                raise AssistAIError("Таймаут ожидания генерации видео.")

            content_fallback = (
                f"{video_provider.base_url.rstrip('/')}/{_VIDEOS_PATH}/{job_id}/content"
            )
            content_candidate = None
            unsigned = status_payload.get("unsigned_urls")
            if isinstance(unsigned, list) and unsigned:
                first = unsigned[0]
                if isinstance(first, str):
                    content_candidate = first
            content_url, content_with_auth, content_pins = resolve_video_resource_url(
                content_candidate,
                provider_base_url=video_provider.base_url,
                fallback=content_fallback,
            )
            try:
                response = await self._video_get(
                    video_provider,
                    content_url,
                    with_auth=content_with_auth,
                    pinned_ips=content_pins,
                    timeout=httpx.Timeout(
                        max(self.config.timeout, _VIDEO_READ_TIMEOUT),
                        connect=self.config.timeout,
                    ),
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                logger.warning("generate_video_download_failed", error=str(exc))
                raise AssistAIError(video_user_error("download")) from exc
            data = response.content
            if not data:
                raise AssistAIError("ProxyAPI вернул пустой видеофайл.")

            await cost_tracker.record(kind="video", amount_usd=VIDEO_USD)
            logger.info("generate_video_done", job_id=job_id, bytes=len(data))
            return VideoResult(data=data, filename="video.mp4", mime_type="video/mp4")
        finally:
            await cost_tracker.release_reservation()

    async def _video_get(
        self,
        provider: ProviderConfig,
        url: str,
        *,
        with_auth: bool,
        timeout: httpx.Timeout,
        pinned_ips: tuple[str, ...] = (),
    ) -> httpx.Response:
        """GET video status/content; never send provider Bearer to foreign hosts.

        Foreign CDN fetches connect to a pinned public IP with Host/SNI of the
        original hostname so a DNS rebind after validation cannot redirect TCP.
        """
        if with_auth:
            return await self._client_for(provider).get(url, timeout=timeout)
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").strip()
        request_url = url
        headers = {"User-Agent": "AssistAIClient/1.0"}
        extensions: dict[str, Any] = {}
        if pinned_ips and hostname:
            request_url = _url_with_pinned_ip(url, pinned_ips[0])
            headers["Host"] = hostname
            extensions["sni_hostname"] = hostname
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=False,
            verify=_ssl_context(),
            headers=headers,
        ) as anon:
            return await anon.get(request_url, extensions=extensions or None)

    async def transcribe_audio(
        self,
        audio_bytes: bytes,
        filename: str = "voice.ogg",
        model: str | None = None,
        language: str | None = None,
    ) -> str:
        """Распознаёт речь через ProxyAPI ``/audio/transcriptions``.

        Args:
            audio_bytes: байты аудиофайла.
            filename: имя файла с расширением. Telegram voice обычно OGG/Opus.
            model: модель распознавания. Если None — ``default_transcribe_model``.
            language: язык речи, например ``ru``. Если None — автоопределение.

        Returns:
            Распознанный текст. Если речи нет — пустая строка.

        Raises:
            ValueError: если ``audio_bytes`` пустой.
            AssistAIError: ошибка HTTP, таймаут, некорректный ответ ProxyAPI.
        """
        if not audio_bytes:
            raise ValueError("audio_bytes не может быть пустым.")
        if self._closed:
            raise AssistAIError("Клиент уже закрыт.")

        selected_model = model or self.config.default_transcribe_model
        audio_provider = self._endpoint("audio")
        url = f"{audio_provider.base_url.rstrip('/')}/{_TRANSCRIBE_PATH}"
        form: dict[str, str] = {"model": selected_model}
        if language:
            form["language"] = language
        mime_type = _audio_mime(filename)
        files = {"file": (filename, audio_bytes, mime_type)}
        # Базовый клиент шлёт Content-Type: application/json. Для multipart
        # нужен отдельный клиент без этого заголовка, иначе boundary пропадёт.
        read_timeout = max(self.config.timeout, _TRANSCRIBE_READ_TIMEOUT)
        timeout = httpx.Timeout(read_timeout, connect=self.config.timeout)
        logger.info(
            "transcribe_audio: модель=%s, байт=%s, язык=%s",
            selected_model,
            len(audio_bytes),
            language or "auto",
        )
        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=False,
                verify=_ssl_context(),
                headers={
                    "Authorization": f"Bearer {audio_provider.api_key}",
                    "User-Agent": "AssistAIClient/1.0",
                },
            ) as client:
                response = await client.post(url, data=form, files=files)
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = _response_error_text(exc.response)
            logger.error(
                "ProxyAPI HTTP %s для %s: %s",
                exc.response.status_code,
                _TRANSCRIBE_PATH,
                detail,
            )
            raise AssistAIError(
                f"ProxyAPI вернул HTTP {exc.response.status_code}: {detail}",
                status_code=exc.response.status_code,
            ) from exc
        except httpx.TimeoutException as exc:
            logger.error("Таймаут запроса к %s", _TRANSCRIBE_PATH)
            raise AssistAIError(
                f"Превышено время ожидания ответа ProxyAPI ({_TRANSCRIBE_PATH})."
            ) from exc
        except httpx.RequestError as exc:
            logger.error("Сетевая ошибка при запросе к %s: %s", _TRANSCRIBE_PATH, exc)
            raise AssistAIError(
                f"Сетевая ошибка при обращении к ProxyAPI: {exc}"
            ) from exc

        try:
            data = response.json()
        except ValueError as exc:
            logger.error("ProxyAPI вернул не-JSON для %s", _TRANSCRIBE_PATH)
            raise AssistAIError("ProxyAPI вернул ответ не в формате JSON.") from exc
        if not isinstance(data, dict):
            raise AssistAIError("ProxyAPI вернул неожиданный формат ответа.")
        text = data.get("text")
        if not isinstance(text, str):
            return ""
        recognized = text.strip()
        logger.info("transcribe_audio: символов=%s", len(recognized))
        return recognized

    async def synthesize_speech(
        self,
        text: str,
        model: str | None = None,
        voice: str | None = None,
        response_format: str | None = None,
        role: str | None = None,
    ) -> SpeechResult:
        """Синтезирует речь: SpeechKit или ProxyAPI ``/audio/speech``.

        Args:
            text: текст ответа ассистента.
            model: TTS-модель (только ProxyAPI).
            voice: голос.
            response_format: формат аудио. По умолчанию opus.
            role: амплуа SpeechKit (игнорируется, если голос его не поддерживает).

        Returns:
            SpeechResult с байтами аудио, filename и mime_type.

        Raises:
            ValueError: пустой текст.
            AssistAIError: ошибка HTTP, таймаут или пустой аудиоответ.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text не может быть пустым.")
        if self._closed:
            raise AssistAIError("Клиент уже закрыт.")

        selected_model = model or self.config.default_tts_model
        from app.llm.models_config import ELEVENLABS_TTS_VOICES, is_elevenlabs_tts_model
        from app.llm.speechkit_tts import is_speechkit_tts_model

        if is_speechkit_tts_model(selected_model):
            return await self._synthesize_speech_speechkit(text, voice=voice, role=role)

        selected_voice = voice or self.config.default_tts_voice
        selected_format = normalize_tts_format(
            response_format or self.config.default_tts_format
        )
        if is_elevenlabs_tts_model(selected_model):
            selected_format = "mp3"
            if selected_voice not in ELEVENLABS_TTS_VOICES:
                selected_voice = ELEVENLABS_TTS_VOICES[0]
        payload = {
            "model": selected_model,
            "input": text,
            "voice": selected_voice,
            "response_format": selected_format,
        }
        speech_provider = self._endpoint("audio")
        url = f"{speech_provider.base_url.rstrip('/')}/{_SPEECH_PATH}"
        read_timeout = max(self.config.timeout, _SPEECH_READ_TIMEOUT)
        timeout = httpx.Timeout(read_timeout, connect=self.config.timeout)
        logger.info(
            "synthesize_speech: модель=%s, голос=%s, формат=%s, символов=%s",
            selected_model,
            selected_voice,
            selected_format,
            len(text),
        )
        try:
            response = await self._client_for(speech_provider).post(
                url, json=payload, timeout=timeout
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = _response_error_text(exc.response)
            logger.error(
                "ProxyAPI HTTP %s для %s: %s",
                exc.response.status_code,
                _SPEECH_PATH,
                detail,
            )
            raise AssistAIError(
                f"ProxyAPI вернул HTTP {exc.response.status_code}: {detail}",
                status_code=exc.response.status_code,
            ) from exc
        except httpx.TimeoutException as exc:
            logger.error("Таймаут запроса к %s", _SPEECH_PATH)
            raise AssistAIError(
                f"Превышено время ожидания ответа ProxyAPI ({_SPEECH_PATH})."
            ) from exc
        except httpx.RequestError as exc:
            logger.error("Сетевая ошибка при запросе к %s: %s", _SPEECH_PATH, exc)
            raise AssistAIError(
                f"Сетевая ошибка при обращении к ProxyAPI: {exc}"
            ) from exc

        audio = response.content
        if not audio:
            raise AssistAIError("ProxyAPI вернул пустой аудиоответ.")
        filename, mime_type = _speech_file(selected_format)
        logger.info("synthesize_speech: байт=%s", len(audio))
        return SpeechResult(audio=audio, filename=filename, mime_type=mime_type)

    async def _synthesize_speech_speechkit(
        self,
        text: str,
        *,
        voice: str | None = None,
        role: str | None = None,
    ) -> SpeechResult:
        """Озвучка через Yandex SpeechKit TTS v3 (OGG Opus)."""
        from app.llm.speechkit_tts import SpeechKitConfig, SpeechKitError, synthesize

        if not self.config.speechkit_api_key:
            raise AssistAIError(
                "Выбрана модель Yandex SpeechKit, но YC_SPEECHKIT_API_KEY не задан."
            )
        sk_config = SpeechKitConfig(
            api_key=self.config.speechkit_api_key,
            folder_id=self.config.speechkit_folder_id,
            voice=self.config.default_tts_voice,
            role=self.config.speechkit_role,
            speed=self.config.speechkit_speed,
            url=self.config.speechkit_url,
        )
        read_timeout = max(self.config.timeout, _SPEECH_READ_TIMEOUT)
        try:
            audio = await synthesize(
                text,
                config=sk_config,
                voice=voice,
                role=role,
                http=self._http,
                timeout=read_timeout,
            )
        except SpeechKitError as exc:
            raise AssistAIError(exc.message, status_code=exc.status_code) from exc
        filename, mime_type = _speech_file("opus")
        logger.info("synthesize_speech speechkit: байт=%s", len(audio))
        return SpeechResult(audio=audio, filename=filename, mime_type=mime_type)
