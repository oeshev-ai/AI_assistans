"""Аудио/видео API: STT, TTS, video generation."""

from __future__ import annotations

import asyncio
import base64
import ipaddress
import re
import socket
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar
from urllib.parse import quote, urlparse, urlunparse

import httpx
import structlog

from app.infra.cost_tracker import VIDEO_USD, cost_tracker, extract_cost_rub
from app.infra.metrics import note_provider_fallback
from app.infra.security import security
from app.llm import polza_media
from app.llm.errors import AssistAIError, fallback_reason
from app.llm.models_config import resolve_polza_video_model
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
    _provider_label,
    _request_base,
    _response_error_text,
    _ssl_context,
)

_T = TypeVar("_T")

logger = structlog.get_logger("AssistAI.media")

_VIDEO_FAILED_USER = (
    "Видео не получилось, в чат оно не придёт. "
    "Если попросить ещё раз, начнётся новая генерация."
)
_VIDEO_POLL_USER = (
    "Видео в этот чат не придёт: задача ушла в сервис, "
    "но готовность проверить не удалось. "
    "Если попросить ещё раз, начнётся новая генерация."
)
_VIDEO_DOWNLOAD_USER = (
    "Видео в этот чат не придёт: файл не скачался. "
    "Если попросить ещё раз, начнётся новая генерация."
)
_VIDEO_RETRY_PAUSE = 1.0


def _video_poll_retryable(exc: httpx.HTTPError) -> bool:
    """Сеть, 429 и 5xx не отменяют уже оплаченную задачу."""
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or code >= 500
    return True


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
    cost_usd: float | None = None
    cost_rub: float | None = None
    cost_source: str | None = None
    provider_id: str | None = None
    model: str | None = None


@dataclass(frozen=True)
class TranscriptionResult:
    """Распознанный текст и фактическая цена, если провайдер её вернул."""

    text: str
    cost_usd: float | None = None
    cost_rub: float | None = None
    cost_source: str | None = None
    provider_id: str | None = None
    model: str | None = None


def _usage_price(
    usage: dict[str, Any] | None,
    rub_per_usd: float,
) -> tuple[float | None, float | None, str | None]:
    """``(cost_usd, cost_rub, cost_source)`` из ``usage`` или пустые поля."""
    rub = extract_cost_rub(usage)
    if rub is None or rub_per_usd <= 0:
        return None, None, None
    return rub / rub_per_usd, rub, "cost_rub"


@dataclass(frozen=True)
class VideoResult:
    """Готовый ролик: байты файла, имя и MIME."""

    data: bytes
    filename: str
    mime_type: str = "video/mp4"
    cost_usd: float | None = None
    cost_rub: float | None = None
    cost_source: str | None = None
    provider_id: str | None = None
    model: str | None = None


# Seedance 2.0: до 9 изображений-референсов и на Polza, и на ProxyAPI.
VIDEO_MAX_REFERENCE_IMAGES = 9
VIDEO_REFERENCE_MAX_BYTES = 10 * 1024 * 1024
_IMAGE_MIME = {
    "image/jpeg": "image/jpeg",
    "image/jpg": "image/jpeg",
    "image/png": "image/png",
    "image/webp": "image/webp",
    "image/gif": "image/gif",
}


@dataclass(frozen=True)
class VideoReference:
    """Картинка-референс для видео: байты или публичный https URL."""

    data: bytes | None = None
    url: str | None = None
    mime_type: str = "image/png"


def safe_image_mime(mime: str | None) -> str:
    """MIME для data-URI референса. Неизвестный тип становится ``image/png``."""
    cleaned = (mime or "").split(";", 1)[0].strip().casefold()
    return _IMAGE_MIME.get(cleaned, "image/png")


def public_https_image_url(url: str) -> bool:
    """True для https-ссылки, которую можно отдать видеосервису как референс."""
    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return False
    if parsed.scheme != "https" or not parsed.hostname:
        return False
    return not _host_is_non_public(parsed.hostname)


def decode_image_b64(value: str) -> tuple[bytes, str]:
    """Сырой base64 или data-URI → байты и MIME. Пустой результат — битые данные."""
    mime = "image/png"
    raw = value.strip()
    if raw.startswith("data:"):
        header, _, payload = raw.partition(",")
        declared = header[5:].split(";", 1)[0].strip()
        if declared:
            mime = safe_image_mime(declared)
        raw = payload
    try:
        data = base64.b64decode(raw, validate=False)
    except Exception:
        return b"", mime
    return data, mime


def checked_video_references(
    references: list[VideoReference] | None,
) -> list[VideoReference]:
    """Проверяет референсы до платного create. Ошибка не открывает fallback."""
    if not references:
        return []
    if len(references) > VIDEO_MAX_REFERENCE_IMAGES:
        raise AssistAIError(
            f"Можно передать не больше {VIDEO_MAX_REFERENCE_IMAGES} референсов.",
            error_code="BAD_REQUEST",
        )
    checked: list[VideoReference] = []
    for ref in references:
        if ref.url:
            url = ref.url.strip()
            if not public_https_image_url(url):
                raise AssistAIError(
                    "Референс-картинка недоступна: нужна публичная https-ссылка.",
                    error_code="BAD_REQUEST",
                )
            checked.append(
                VideoReference(url=url, mime_type=safe_image_mime(ref.mime_type))
            )
            continue
        if not ref.data:
            raise AssistAIError("Референс-картинка пустая.", error_code="BAD_REQUEST")
        if len(ref.data) > VIDEO_REFERENCE_MAX_BYTES:
            raise AssistAIError(
                "Референс-картинка слишком большая для запроса видео.",
                error_code="PAYLOAD_TOO_LARGE",
            )
        checked.append(
            VideoReference(data=ref.data, mime_type=safe_image_mime(ref.mime_type))
        )
    return checked


def polza_reference_images(references: list[VideoReference]) -> list[dict[str, str]]:
    """``input.images`` для Polza Seedance: url или сырой base64."""
    images: list[dict[str, str]] = []
    for ref in references:
        if ref.url:
            images.append({"type": "url", "data": ref.url})
        elif ref.data:
            images.append(
                {
                    "type": "base64",
                    "data": base64.b64encode(ref.data).decode("ascii"),
                }
            )
    return images


def proxy_input_references(references: list[VideoReference]) -> list[dict[str, Any]]:
    """``input_references`` ProxyAPI. Байты уходят data-URI: своего хранилища нет."""
    items: list[dict[str, Any]] = []
    for ref in references:
        if ref.url:
            url = ref.url
        elif ref.data:
            mime = safe_image_mime(ref.mime_type)
            payload = base64.b64encode(ref.data).decode("ascii")
            url = f"data:{mime};base64,{payload}"
        else:
            continue
        items.append({"type": "image_url", "image_url": {"url": url}})
    return items


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


def _trusted_video_fallback(
    candidate: str | None, fallback: str
) -> tuple[str, bool, tuple[str, ...]]:
    """Return a trusted provider fallback; never re-auth a rejected candidate.

    Callers must pass a provider-hosted HTTPS URL as ``fallback``. If ``fallback``
    is empty or identical to the rejected ``candidate``, returns an empty URL so
    the caller fails closed instead of SSRF / key-leak via ``with_auth=True``.
    """
    fb = (fallback or "").strip()
    cand = candidate.strip() if isinstance(candidate, str) else ""
    if not fb or fb == cand:
        return "", False, ()
    return fb, True, ()


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
      trusted ``fallback`` (never the rejected candidate itself).
    """
    if not isinstance(candidate, str) or not candidate.strip():
        return _trusted_video_fallback(candidate, fallback)
    raw = candidate.strip()
    try:
        parsed = urlparse(raw)
        provider = urlparse(provider_base_url)
    except ValueError:
        return _trusted_video_fallback(raw, fallback)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return _trusted_video_fallback(raw, fallback)
    host = parsed.hostname.casefold()
    provider_host = (provider.hostname or "").casefold()
    if host == provider_host:
        # Bearer must not travel over plaintext, even to the provider host.
        if parsed.scheme != "https":
            return _trusted_video_fallback(raw, fallback)
        return raw, True, ()
    if _host_is_non_public(host):
        return _trusted_video_fallback(raw, fallback)
    pinned = _public_resolved_ips(host)
    if not pinned:
        return _trusted_video_fallback(raw, fallback)
    if parsed.scheme != "https":
        return _trusted_video_fallback(raw, fallback)
    return raw, False, pinned


_POLZA_CDN_HOSTS = frozenset({"s3.polza.ai", "s3-alt.polza.ai"})
_POLZA_CDN_MIRRORS: dict[str, tuple[str, ...]] = {
    "s3.polza.ai": ("s3-alt.polza.ai",),
    "s3-alt.polza.ai": ("s3.polza.ai",),
}


_PAID_JOB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


def paid_job_id(value: str) -> bool:
    """Id задачи Polza или ProxyAPI, который можно снова запросить."""
    return isinstance(value, str) and _PAID_JOB_ID.fullmatch(value) is not None


def polza_cdn_url(url: str) -> bool:
    """Публичный https на CDN Polza. Файл там живёт 7 дней."""
    if not isinstance(url, str) or not public_https_image_url(url):
        return False
    host = (urlparse(url.strip()).hostname or "").casefold()
    return host in _POLZA_CDN_HOSTS


def cdn_download_candidates(url: str) -> list[str]:
    """Сначала зеркало CDN, затем исходный URL. Чужой хост — только он сам."""
    raw = url.strip()
    parsed = urlparse(raw)
    host = (parsed.hostname or "").casefold()
    ordered = [
        urlunparse(parsed._replace(netloc=mirror))
        for mirror in _POLZA_CDN_MIRRORS.get(host, ())
    ]
    if raw not in ordered:
        ordered.append(raw)
    return ordered


async def download_public_video(
    url: str, *, timeout: httpx.Timeout, attempts: int = 2
) -> bytes:
    """Скачивает уже готовый ролик с публичного CDN.

    ``s3.polza.ai`` с этой сети часто не открывается, зеркало отдаёт тот же
    объект. Если оба хоста молчат, ошибка несёт ``pending_url`` — бот повторит
    скачивание без новой генерации.
    """
    candidates = cdn_download_candidates(url) if isinstance(url, str) else []
    last_code = "NETWORK_ERROR"
    permanent = False
    attempted = False
    for candidate in candidates:
        resolved, with_auth, pins = resolve_video_resource_url(
            candidate,
            provider_base_url="https://polza.ai",
            fallback="",
        )
        if not resolved or with_auth:
            logger.warning(
                "video_cdn_url_rejected",
                host=urlparse(candidate).hostname if isinstance(candidate, str) else "",
            )
            continue
        attempted = True
        try:
            data = await _fetch_public_video(resolved, pins, timeout, attempts=attempts)
        except AssistAIError as exc:
            last_code = exc.error_code or last_code
            if exc.error_code == "NOT_FOUND" and candidate == url.strip():
                permanent = True
            logger.warning(
                "video_cdn_download_failed",
                host=urlparse(candidate).hostname,
                error_code=exc.error_code,
            )
            continue
        if data:
            if candidate != url.strip():
                logger.info(
                    "video_cdn_fallback_used",
                    host=urlparse(candidate).hostname,
                )
            return data
    keep = isinstance(url, str) and polza_cdn_url(url) and not permanent
    if not attempted and not keep:
        last_code = "GUARD_UNSAFE_URL"
    raise AssistAIError(
        video_user_error("download"),
        error_code=last_code,
        job_committed=True,
        pending_url=url.strip() if keep and isinstance(url, str) else None,
    )


async def _fetch_public_video(
    url: str,
    pinned_ips: tuple[str, ...],
    timeout: httpx.Timeout,
    *,
    attempts: int = 2,
) -> bytes:
    """GET без ключа провайдера. Приватный адрес сюда не доходит."""
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").strip()
    request_url = url
    headers = {"User-Agent": "AssistAIClient/1.0"}
    extensions: dict[str, Any] = {}
    if pinned_ips and hostname:
        request_url = _url_with_pinned_ip(url, pinned_ips[0])
        headers["Host"] = hostname
        extensions["sni_hostname"] = hostname
    last: httpx.HTTPError | None = None
    tries = max(1, attempts)
    for attempt in range(1, tries + 1):
        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=False,
                verify=_ssl_context(),
                headers=headers,
            ) as anon:
                response = await anon.get(request_url, extensions=extensions or None)
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            if code in {403, 404, 410}:
                raise AssistAIError(
                    video_user_error("download"),
                    error_code="NOT_FOUND",
                    job_committed=True,
                ) from exc
            last = exc
        except httpx.HTTPError as exc:
            last = exc
        else:
            if response.content:
                return bytes(response.content)
        if attempt < tries:
            logger.warning("generate_video_download_retry", attempt=attempt)
            await asyncio.sleep(_VIDEO_RETRY_PAUSE)
    raise AssistAIError(
        video_user_error("download"),
        error_code="NETWORK_ERROR",
        job_committed=True,
    ) from last


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

    async def _video_with_fallback(
        self,
        attempt: Callable[[ProviderConfig], Awaitable[_T]],
    ) -> _T:
        """Видео у основного провайдера; один fallback только на create-time сбой.

        После принятия задачи провайдером (``AssistAIError.job_committed``) fallback
        запрещён — иначе вторая платная job на ProxyAPI.
        """
        primary, fallback = self._registry.resolve_route("video")
        try:
            return await attempt(primary)
        except AssistAIError as exc:
            reason = fallback_reason(exc)
            if reason == "insufficient_balance":
                logger.error("provider_balance_exhausted", provider=primary.id)
            if fallback is None or reason is None:
                raise
            logger.warning(
                "provider_fallback",
                **{"from": primary.id, "to": fallback.id, "reason": reason},
            )
            note_provider_fallback(primary.id, fallback.id, reason)
            return await attempt(fallback)

    async def generate_video(
        self,
        prompt: str,
        *,
        model: str | None = None,
        duration: int | None = None,
        resolution: str | None = None,
        references: list[VideoReference] | None = None,
    ) -> VideoResult:
        """Создаёт видео-задачу, ждёт completion и скачивает mp4."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt не может быть пустым.")
        if self._closed:
            raise AssistAIError("Клиент уже закрыт.")

        catalog_model = model or self.config.default_video_model
        sanitized_prompt = security.sanitize_prompt(prompt.strip())
        create_timeout = httpx.Timeout(
            max(self.config.timeout, _VIDEO_READ_TIMEOUT),
            connect=self.config.timeout,
        )

        await cost_tracker.check_budget(estimated=VIDEO_USD)
        try:
            checked_refs = checked_video_references(references)

            async def attempt(provider: ProviderConfig) -> VideoResult:
                if provider.kind == "polza_media":
                    return await self._generate_video_polza(
                        provider,
                        catalog_model=catalog_model,
                        prompt=sanitized_prompt,
                        duration=duration,
                        resolution=resolution,
                        references=checked_refs,
                        timeout=create_timeout,
                    )
                return await self._generate_video_proxyapi(
                    provider,
                    catalog_model=catalog_model,
                    prompt=sanitized_prompt,
                    duration=duration,
                    resolution=resolution,
                    references=checked_refs,
                    timeout=create_timeout,
                )

            logger.info(
                "generate_video_create",
                model=catalog_model,
                prompt_chars=len(sanitized_prompt),
                references=len(checked_refs),
            )
            result = await self._video_with_fallback(attempt)
            amount = result.cost_usd if result.cost_usd is not None else VIDEO_USD
            await cost_tracker.record(
                kind="video",
                amount_usd=amount,
                provider=result.provider_id,
                model=result.model or catalog_model,
                source=result.cost_source or "flat",
                cost_rub=result.cost_rub,
            )
            logger.info("generate_video_done", bytes=len(result.data))
            return result
        finally:
            await cost_tracker.release_reservation()

    def _video_result(
        self,
        provider: ProviderConfig,
        *,
        data: bytes,
        model: str,
        cost_rub: float | None = None,
        filename: str = "video.mp4",
        mime_type: str = "video/mp4",
    ) -> VideoResult:
        """Assemble ``VideoResult`` with optional Polza ``cost_rub`` billing."""
        cost_usd: float | None = None
        cost_source: str | None = None
        if cost_rub is not None and self.config.rub_per_usd > 0:
            cost_usd = cost_rub / self.config.rub_per_usd
            cost_source = "cost_rub"
        return VideoResult(
            data=data,
            filename=filename,
            mime_type=mime_type,
            cost_usd=cost_usd,
            cost_rub=cost_rub if cost_source else None,
            cost_source=cost_source,
            provider_id=provider.id,
            model=model,
        )

    async def _generate_video_polza(
        self,
        provider: ProviderConfig,
        *,
        catalog_model: str,
        prompt: str,
        duration: int | None,
        resolution: str | None,
        references: list[VideoReference],
        timeout: httpx.Timeout,
    ) -> VideoResult:
        polza_model = resolve_polza_video_model(catalog_model) or catalog_model
        # Video jobs often exceed the image poll budget; floor at ProxyAPI poll max.
        total_wait = max(float(self.config.polza_total_wait), _VIDEO_POLL_MAX_SECONDS)
        media = await polza_media.generate_video(
            self._client_for(provider),
            provider,
            model=polza_model,
            prompt=prompt,
            duration=duration,
            resolution=resolution,
            images=polza_reference_images(references) or None,
            poll_interval=max(
                float(self.config.polza_poll_interval), _VIDEO_POLL_INTERVAL
            ),
            total_wait=total_wait,
            timeout=timeout,
        )
        # Media job already completed/billed on Polza — download failures stay put.
        first = media.images[0]
        if first.get("type") == "base64":
            raw = first.get("value") or ""
            try:
                data = base64.b64decode(raw, validate=False)
            except Exception as exc:
                raise AssistAIError(
                    video_user_error("download"),
                    error_code="BAD_GATEWAY",
                    job_committed=True,
                ) from exc
            if not data:
                raise AssistAIError(
                    video_user_error("download"),
                    error_code="BAD_GATEWAY",
                    job_committed=True,
                )
            return self._video_result(
                provider, data=data, model=polza_model, cost_rub=media.cost_rub
            )

        content_url = first.get("value") or ""
        if not content_url:
            raise AssistAIError(
                video_user_error("download"),
                error_code="SERVICE_UNAVAILABLE",
                job_committed=True,
            )
        try:
            data = await download_public_video(
                content_url,
                timeout=httpx.Timeout(45.0, connect=15.0),
                attempts=1,
            )
        except AssistAIError as exc:
            if media.media_id:
                exc.pending_provider = "polza"
                exc.pending_job_id = media.media_id
                exc.pending_kind = "video"
            raise
        return self._video_result(
            provider, data=data, model=polza_model, cost_rub=media.cost_rub
        )

    async def _generate_video_proxyapi(
        self,
        provider: ProviderConfig,
        *,
        catalog_model: str,
        prompt: str,
        duration: int | None,
        resolution: str | None,
        references: list[VideoReference],
        timeout: httpx.Timeout,
    ) -> VideoResult:
        label = _provider_label(provider)
        payload: dict[str, Any] = {
            "model": catalog_model,
            "prompt": prompt,
        }
        if isinstance(duration, int) and duration > 0:
            payload["duration"] = duration
        if isinstance(resolution, str) and resolution.strip():
            payload["resolution"] = resolution.strip()
        image_refs = proxy_input_references(references)
        if image_refs:
            payload["input_references"] = image_refs

        created = await self._post_json(
            _VIDEOS_PATH, payload, timeout=timeout, provider=provider
        )
        # Create succeeded — poll/download errors must not open another provider.
        job_id: str | None = None
        recoverable = True
        try:
            job_id = created.get("id") if isinstance(created.get("id"), str) else None
            if not job_id:
                raise AssistAIError(
                    f"{label} не вернул id видео-задачи.",
                    error_code="BAD_GATEWAY",
                    job_committed=True,
                )
            status_fallback = f"{provider.base_url.rstrip('/')}/{_VIDEOS_PATH}/{job_id}"
            status_url, status_with_auth, status_pins = resolve_video_resource_url(
                created.get("polling_url")
                if isinstance(created.get("polling_url"), str)
                else None,
                provider_base_url=provider.base_url,
                fallback=status_fallback,
            )
            if not status_url:
                raise AssistAIError(
                    video_user_error("poll"),
                    error_code="BAD_GATEWAY",
                    job_committed=True,
                )

            deadline = time.monotonic() + _VIDEO_POLL_MAX_SECONDS
            status_payload: dict[str, Any] = created
            while time.monotonic() < deadline:
                status = str(status_payload.get("status") or "").casefold()
                if status in {"completed", "succeeded", "success"}:
                    break
                if status in {"failed", "error", "cancelled"}:
                    recoverable = False
                    logger.warning(
                        "generate_video_failed",
                        status=status,
                        job_id=job_id,
                        payload_keys=sorted(status_payload.keys()),
                    )
                    raise AssistAIError(
                        video_user_error("failed"),
                        error_code="INTERNAL_ERROR",
                        job_committed=True,
                    )
                await asyncio.sleep(_VIDEO_POLL_INTERVAL)
                try:
                    response = await self._video_get(
                        provider,
                        status_url,
                        with_auth=status_with_auth,
                        pinned_ips=status_pins,
                        timeout=httpx.Timeout(30.0, connect=self.config.timeout),
                    )
                    response.raise_for_status()
                    body = response.json()
                except httpx.HTTPError as exc:
                    if not _video_poll_retryable(exc):
                        logger.warning(
                            "generate_video_poll_failed",
                            error=str(exc),
                            job_id=job_id,
                        )
                        raise AssistAIError(
                            video_user_error("poll"),
                            error_code="NETWORK_ERROR",
                            job_committed=True,
                        ) from exc
                    logger.warning(
                        "generate_video_poll_retry",
                        error=str(exc),
                        job_id=job_id,
                    )
                    continue
                except ValueError:
                    logger.warning(
                        "generate_video_poll_retry", reason="bad_json", job_id=job_id
                    )
                    continue
                if not isinstance(body, dict):
                    logger.warning(
                        "generate_video_poll_retry", reason="bad_body", job_id=job_id
                    )
                    continue
                status_payload = body
            else:
                raise AssistAIError(
                    video_user_error("poll"),
                    error_code="POLL_TIMEOUT",
                    job_committed=True,
                )

            content_fallback = (
                f"{provider.base_url.rstrip('/')}/{_VIDEOS_PATH}/{job_id}/content"
            )
            content_candidate = None
            unsigned = status_payload.get("unsigned_urls")
            if isinstance(unsigned, list) and unsigned:
                first = unsigned[0]
                if isinstance(first, str):
                    content_candidate = first
            content_url, content_with_auth, content_pins = resolve_video_resource_url(
                content_candidate,
                provider_base_url=provider.base_url,
                fallback=content_fallback,
            )
            if not content_url:
                raise AssistAIError(
                    video_user_error("download"),
                    error_code="BAD_GATEWAY",
                    job_committed=True,
                )
            data = await self._download_video_bytes(
                provider,
                content_url,
                with_auth=content_with_auth,
                pinned_ips=content_pins,
                timeout=timeout,
            )
            if not data:
                raise AssistAIError(
                    f"{label} вернул пустой видеофайл.",
                    error_code="BAD_GATEWAY",
                    job_committed=True,
                )
            return self._video_result(provider, data=data, model=catalog_model)
        except AssistAIError as exc:
            exc.job_committed = True
            if recoverable and job_id:
                exc.pending_provider = provider.id
                exc.pending_job_id = job_id
                exc.pending_kind = "video"
            raise

    async def _download_video_bytes(
        self,
        provider: ProviderConfig,
        url: str,
        *,
        with_auth: bool,
        pinned_ips: tuple[str, ...],
        timeout: httpx.Timeout,
    ) -> bytes:
        """Несколько попыток скачать уже оплаченный ролик."""
        last: httpx.HTTPError | None = None
        for attempt in range(1, 4):
            try:
                response = await self._video_get(
                    provider,
                    url,
                    with_auth=with_auth,
                    pinned_ips=pinned_ips,
                    timeout=timeout,
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                last = exc
                if attempt == 3 or not _video_poll_retryable(exc):
                    logger.warning("generate_video_download_failed", error=str(exc))
                    raise AssistAIError(
                        video_user_error("download"),
                        error_code="NETWORK_ERROR",
                        job_committed=True,
                    ) from exc
                logger.warning(
                    "generate_video_download_retry",
                    attempt=attempt,
                    error=str(exc),
                )
                await asyncio.sleep(_VIDEO_RETRY_PAUSE)
                continue
            data = response.content
            if data:
                return data
        logger.warning(
            "generate_video_download_failed",
            error=str(last) if last else "empty",
        )
        return b""

    async def recover_paid_file(
        self, provider_id: str, job_id: str, *, kind: str
    ) -> bytes:
        """Забирает уже оплаченный файл по id задачи. ``NOT_READY`` — ещё не готов."""
        if provider_id not in {"polza", "proxyapi"} or not paid_job_id(job_id):
            raise AssistAIError(
                video_user_error("failed") if kind == "video" else "Файл не найден.",
                error_code="BAD_REQUEST",
                job_committed=True,
            )
        try:
            provider = self._registry.get(provider_id)
        except KeyError as exc:
            raise AssistAIError(
                video_user_error("failed") if kind == "video" else "Файл не найден.",
                error_code="FAILED",
                job_committed=True,
            ) from exc
        if provider_id == "polza":
            return await self._recover_polza_file(provider, job_id, kind=kind)
        if kind != "video":
            raise AssistAIError(
                "Файл не найден.",
                error_code="FAILED",
                job_committed=True,
            )
        return await self._recover_proxy_video(provider, job_id)

    async def _recover_polza_file(
        self, provider: ProviderConfig, job_id: str, *, kind: str
    ) -> bytes:
        url = f"{provider.base_url.rstrip('/')}/v1/media/{quote(job_id, safe='')}"
        timeout = httpx.Timeout(30.0, connect=15.0)
        try:
            response = await self._client_for(provider).get(url, timeout=timeout)
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise AssistAIError(
                video_user_error("poll")
                if kind == "video"
                else "Файл ещё не скачался.",
                error_code="NOT_READY",
                job_committed=True,
            ) from exc
        if not isinstance(body, dict):
            raise AssistAIError(
                video_user_error("poll"),
                error_code="NOT_READY",
                job_committed=True,
            )
        status = str(body.get("status") or "").casefold()
        logger.info("paid_file_status", provider="polza", job_id=job_id, status=status)
        if status in polza_media._PENDING_STATUSES:
            raise AssistAIError(
                video_user_error("poll") if kind == "video" else "Файл ещё не готов.",
                error_code="NOT_READY",
                job_committed=True,
            )
        if status in polza_media._FAILED_STATUSES:
            raise AssistAIError(
                video_user_error("failed")
                if kind == "video"
                else "Картинка не получилась.",
                error_code="FAILED",
                job_committed=True,
            )
        result = polza_media.parse_media_payload(body)
        if not result.images:
            raise AssistAIError(
                video_user_error("download")
                if kind == "video"
                else "Файл ещё не готов.",
                error_code="NOT_READY",
                job_committed=True,
            )
        first = result.images[0]
        if first.get("type") == "base64":
            try:
                data = base64.b64decode(first.get("value") or "", validate=False)
            except Exception as exc:
                raise AssistAIError(
                    video_user_error("download"),
                    error_code="NOT_READY",
                    job_committed=True,
                ) from exc
            if not data:
                raise AssistAIError(
                    video_user_error("download"),
                    error_code="NOT_READY",
                    job_committed=True,
                )
            return data
        content_url = first.get("value") or ""
        return await download_public_video(
            content_url,
            timeout=httpx.Timeout(45.0, connect=15.0),
            attempts=1,
        )

    async def _recover_proxy_video(
        self, provider: ProviderConfig, job_id: str
    ) -> bytes:
        status_url = f"{provider.base_url.rstrip('/')}/{_VIDEOS_PATH}/{job_id}"
        timeout = httpx.Timeout(30.0, connect=15.0)
        try:
            response = await self._video_get(
                provider,
                status_url,
                with_auth=True,
                pinned_ips=(),
                timeout=timeout,
            )
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise AssistAIError(
                video_user_error("poll"),
                error_code="NOT_READY",
                job_committed=True,
            ) from exc
        if not isinstance(body, dict):
            raise AssistAIError(
                video_user_error("poll"),
                error_code="NOT_READY",
                job_committed=True,
            )
        status = str(body.get("status") or "").casefold()
        logger.info(
            "paid_file_status", provider=provider.id, job_id=job_id, status=status
        )
        if status in {"failed", "error", "cancelled"}:
            raise AssistAIError(
                video_user_error("failed"),
                error_code="FAILED",
                job_committed=True,
            )
        if status not in {"completed", "succeeded", "success"}:
            raise AssistAIError(
                video_user_error("poll"),
                error_code="NOT_READY",
                job_committed=True,
            )
        content_url = f"{status_url}/content"
        data = await self._download_video_bytes(
            provider,
            content_url,
            with_auth=True,
            pinned_ips=(),
            timeout=httpx.Timeout(45.0, connect=15.0),
        )
        if not data:
            raise AssistAIError(
                video_user_error("download"),
                error_code="NOT_READY",
                job_committed=True,
            )
        return data

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
        if not isinstance(url, str) or not url.strip():
            raise AssistAIError(video_user_error("download"), error_code="BAD_GATEWAY")
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

    async def _call_with_fallback(
        self,
        primary: ProviderConfig,
        fallback: ProviderConfig | None,
        attempt: Callable[[ProviderConfig], Awaitable[_T]],
    ) -> _T:
        """Один вызов у primary и, если хост недоступен, один у fallback."""
        try:
            return await attempt(primary)
        except AssistAIError as exc:
            reason = fallback_reason(exc)
            if reason == "insufficient_balance":
                logger.error("provider_balance_exhausted", provider=primary.id)
            if fallback is None or reason is None:
                raise
            logger.warning(
                "provider_fallback",
                **{"from": primary.id, "to": fallback.id, "reason": reason},
            )
            note_provider_fallback(primary.id, fallback.id, reason)
            return await attempt(fallback)

    async def _audio_with_fallback(
        self,
        attempt: Callable[[ProviderConfig], Awaitable[_T]],
    ) -> _T:
        """Audio у основного провайдера; один fallback только при недоступности."""
        primary, fallback = self._registry.resolve_route("audio")
        return await self._call_with_fallback(primary, fallback, attempt)

    async def _tts_with_fallback(
        self,
        attempt: Callable[[ProviderConfig], Awaitable[_T]],
    ) -> _T:
        """Каталожный TTS у ``TTS_PROVIDER``; запасной хост — маршрут audio."""
        route_primary, route_fallback = self._registry.resolve_route("audio")
        primary = route_primary
        wanted = self.config.tts_provider
        if wanted in {"polza", "proxyapi"}:
            try:
                primary = self._registry.get(wanted)
            except KeyError:
                primary = route_primary
        fallback = route_fallback
        if fallback is not None and fallback.id == primary.id:
            fallback = route_primary if route_primary.id != primary.id else None
        elif fallback is None and route_primary.id != primary.id:
            fallback = route_primary
        return await self._call_with_fallback(primary, fallback, attempt)

    async def transcribe_audio(
        self,
        audio_bytes: bytes,
        filename: str = "voice.ogg",
        model: str | None = None,
        language: str | None = None,
    ) -> TranscriptionResult:
        """Распознаёт речь через ``/audio/transcriptions`` (AUDIO_PROVIDER).

        Args:
            audio_bytes: байты аудиофайла.
            filename: имя файла с расширением. Telegram voice обычно OGG/Opus.
            model: модель распознавания. Если None — ``default_transcribe_model``.
            language: язык речи, например ``ru``. Если None — автоопределение.

        Returns:
            Текст и, для Polza, фактическая цена из ``usage.cost_rub``.
            Если речи нет — пустой ``text``.

        Raises:
            ValueError: если ``audio_bytes`` пустой.
            AssistAIError: ошибка HTTP, таймаут, некорректный ответ провайдера.
        """
        if not audio_bytes:
            raise ValueError("audio_bytes не может быть пустым.")
        if self._closed:
            raise AssistAIError("Клиент уже закрыт.")

        selected_model = model or self.config.default_transcribe_model
        form: dict[str, str] = {"model": selected_model}
        if language:
            form["language"] = language
        mime_type = _audio_mime(filename)
        files = {"file": (filename, audio_bytes, mime_type)}
        read_timeout = max(self.config.timeout, _TRANSCRIBE_READ_TIMEOUT)
        timeout = httpx.Timeout(read_timeout, connect=self.config.timeout)
        logger.info(
            "transcribe_audio: модель=%s, байт=%s, язык=%s",
            selected_model,
            len(audio_bytes),
            language or "auto",
        )

        async def attempt(provider: ProviderConfig) -> TranscriptionResult:
            label = _provider_label(provider)
            url = f"{_request_base(provider, _TRANSCRIBE_PATH)}/{_TRANSCRIBE_PATH}"
            try:
                # Базовый клиент шлёт Content-Type: application/json. Для multipart
                # нужен отдельный клиент без этого заголовка, иначе boundary пропадёт.
                async with httpx.AsyncClient(
                    timeout=timeout,
                    follow_redirects=False,
                    verify=_ssl_context(),
                    headers={
                        "Authorization": f"Bearer {provider.api_key}",
                        "User-Agent": "AssistAIClient/1.0",
                    },
                ) as client:
                    response = await client.post(url, data=form, files=files)
                    response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                detail = _response_error_text(exc.response)
                logger.error(
                    "audio_http_error",
                    status_code=exc.response.status_code,
                    path=_TRANSCRIBE_PATH,
                    provider=provider.id,
                    detail=detail,
                )
                raise AssistAIError(
                    f"{label} вернул HTTP {exc.response.status_code}: {detail}",
                    status_code=exc.response.status_code,
                ) from exc
            except httpx.TimeoutException as exc:
                logger.error(
                    "audio_timeout", path=_TRANSCRIBE_PATH, provider=provider.id
                )
                raise AssistAIError(
                    f"Превышено время ожидания ответа {label} ({_TRANSCRIBE_PATH}).",
                    error_code="TIMEOUT",
                ) from exc
            except httpx.RequestError as exc:
                logger.error(
                    "audio_network_error",
                    path=_TRANSCRIBE_PATH,
                    provider=provider.id,
                    error=str(exc),
                )
                raise AssistAIError(
                    f"Сетевая ошибка при обращении к {label}: {exc}",
                    error_code="NETWORK_ERROR",
                ) from exc

            try:
                data = response.json()
            except ValueError as exc:
                logger.error(
                    "audio_non_json", path=_TRANSCRIBE_PATH, provider=provider.id
                )
                raise AssistAIError(f"{label} вернул ответ не в формате JSON.") from exc
            if not isinstance(data, dict):
                raise AssistAIError(f"{label} вернул неожиданный формат ответа.")
            text = data.get("text")
            recognized_text = text.strip() if isinstance(text, str) else ""
            usage = data.get("usage")
            cost_usd, cost_rub, cost_source = _usage_price(
                usage if isinstance(usage, dict) else None,
                self.config.rub_per_usd,
            )
            return TranscriptionResult(
                text=recognized_text,
                cost_usd=cost_usd,
                cost_rub=cost_rub,
                cost_source=cost_source,
                provider_id=provider.id,
                model=selected_model,
            )

        recognized = await self._audio_with_fallback(attempt)
        logger.info("transcribe_audio: символов=%s", len(recognized.text))
        return recognized

    async def synthesize_speech(
        self,
        text: str,
        model: str | None = None,
        voice: str | None = None,
        response_format: str | None = None,
        role: str | None = None,
    ) -> SpeechResult:
        """Синтезирует речь: SpeechKit или ``/audio/speech`` у ``TTS_PROVIDER``.

        Args:
            text: текст ответа ассистента.
            model: TTS-модель (Polza/ProxyAPI).
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

        from app.llm.models_config import resolve_tts_model
        from app.llm.speechkit_tts import is_speechkit_tts_model

        raw_model = model or self.config.default_tts_model
        selected_model = resolve_tts_model(raw_model) or raw_model
        if selected_model != raw_model:
            logger.info("tts_model_aliased", old=raw_model, new=selected_model)

        if is_speechkit_tts_model(selected_model):
            return await self._synthesize_speech_speechkit(text, voice=voice, role=role)

        selected_voice = voice or self.config.default_tts_voice
        selected_format = normalize_tts_format(
            response_format or self.config.default_tts_format
        )
        payload = {
            "model": selected_model,
            "input": text,
            "voice": selected_voice,
            "response_format": selected_format,
        }
        read_timeout = max(self.config.timeout, _SPEECH_READ_TIMEOUT)
        timeout = httpx.Timeout(read_timeout, connect=self.config.timeout)
        logger.info(
            "synthesize_speech: модель=%s, голос=%s, формат=%s, символов=%s",
            selected_model,
            selected_voice,
            selected_format,
            len(text),
        )

        async def attempt(provider: ProviderConfig) -> SpeechResult:
            label = _provider_label(provider)
            url = f"{_request_base(provider, _SPEECH_PATH)}/{_SPEECH_PATH}"
            try:
                response = await self._client_for(provider).post(
                    url, json=payload, timeout=timeout
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                detail = _response_error_text(exc.response)
                logger.error(
                    "audio_http_error",
                    status_code=exc.response.status_code,
                    path=_SPEECH_PATH,
                    provider=provider.id,
                    detail=detail,
                )
                raise AssistAIError(
                    f"{label} вернул HTTP {exc.response.status_code}: {detail}",
                    status_code=exc.response.status_code,
                ) from exc
            except httpx.TimeoutException as exc:
                logger.error("audio_timeout", path=_SPEECH_PATH, provider=provider.id)
                raise AssistAIError(
                    f"Превышено время ожидания ответа {label} ({_SPEECH_PATH}).",
                    error_code="TIMEOUT",
                ) from exc
            except httpx.RequestError as exc:
                logger.error(
                    "audio_network_error",
                    path=_SPEECH_PATH,
                    provider=provider.id,
                    error=str(exc),
                )
                raise AssistAIError(
                    f"Сетевая ошибка при обращении к {label}: {exc}",
                    error_code="NETWORK_ERROR",
                ) from exc

            filename, mime_type = _speech_file(selected_format)
            cost_usd: float | None = None
            cost_rub: float | None = None
            cost_source: str | None = None
            if provider.kind == "polza_media":
                try:
                    data = response.json()
                except ValueError as exc:
                    raise AssistAIError(
                        f"{label} вернул ответ не в формате JSON."
                    ) from exc
                if not isinstance(data, dict):
                    raise AssistAIError(f"{label} вернул неожиданный формат ответа.")
                raw_audio = data.get("audio")
                if not isinstance(raw_audio, str) or not raw_audio.strip():
                    raise AssistAIError(f"{label} вернул пустой аудиоответ.")
                try:
                    audio = base64.b64decode(raw_audio, validate=False)
                except Exception as exc:
                    raise AssistAIError(
                        f"{label} вернул некорректный base64 аудио."
                    ) from exc
                if not audio:
                    raise AssistAIError(f"{label} вернул пустой аудиоответ.")
                usage = data.get("usage")
                cost_usd, cost_rub, cost_source = _usage_price(
                    usage if isinstance(usage, dict) else None,
                    self.config.rub_per_usd,
                )
                content_type = data.get("contentType")
                if isinstance(content_type, str) and content_type.strip():
                    mime_type = content_type.strip()
                    if "mpeg" in mime_type or mime_type.endswith("/mp3"):
                        filename = "answer.mp3"
                    elif "ogg" in mime_type or "opus" in mime_type:
                        filename = "answer.ogg"
                    elif "wav" in mime_type:
                        filename = "answer.wav"
            else:
                audio = response.content
                if not audio:
                    raise AssistAIError(f"{label} вернул пустой аудиоответ.")
            return SpeechResult(
                audio=audio,
                filename=filename,
                mime_type=mime_type,
                cost_usd=cost_usd,
                cost_rub=cost_rub,
                cost_source=cost_source,
                provider_id=provider.id,
                model=selected_model,
            )

        speech = await self._tts_with_fallback(attempt)
        logger.info("synthesize_speech: байт=%s", len(speech.audio))
        return speech

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
