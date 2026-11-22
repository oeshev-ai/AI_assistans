"""Адаптер Polza.ai Media API: генерация и правка картинок через ``POST /v1/media``.

Контракт проверен живым прогоном (``app/docs/polza-media-contract.md``):

* для ``openai/gpt-image-2.6-flare`` ``image_resolution`` даёт HTTP 400,
  размер задаётся ``aspect_ratio`` + пиксельным ``size``; ``max_price.image``
  при token pricing — 400 (не отправляем);
* для ``flux-3-image`` / ``flux-3-image-klein`` / ``qwen/image-2.1`` —
  ``aspect_ratio`` + ``image_resolution``; без ``quality`` и без
  ``provider.only=["OpenAI"]``; у klein нет 768/1.5K и ``max_price``;
* ``qwen/qwen-image-2.1-pro`` → алиас ``qwen/image-2.1``;
  ``google/gemini-nano-banana-2.1`` — ``aspect_ratio`` + ``image_resolution``;
  ``recraft/*`` на Polza нет — ``SERVICE_UNAVAILABLE`` для fallback на ProxyAPI;
* ``data`` — список ``[{"url": ...}]`` (допускаем и объект, и ``b64_json``);
* референсы передаются как ``{"type": "base64", "data": ...}``;
* опрос статуса — только ``GET /v1/media/{id}``;
* ``complexity`` и ``background`` не отправляются;
* ``usage.cost_rub`` приходит уже в синхронном ответе.

Адаптер не знает про breaker и fallback: это делает ``assist_ai``.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

import httpx
import structlog

from app.infra.cost_tracker import extract_cost_rub
from app.llm.errors import AssistAIError
from app.llm.providers import ProviderConfig

logger = structlog.get_logger("polza_media")

POLZA_MAX_REFERENCE_IMAGES = 16
_MEDIA_PATH = "/v1/media"
_MAX_USER_LEN = 256
_MAX_POLL_ERRORS = 2

# Соотношения, которые отдаём Polza. Расширено относительно набора FLUX
# (``_snap_aspect_ratio`` в ``assist_ai``), но только значениями, которые
# документация перечисляет явно; экзотику (``3:1``) снапим к ближайшему.
# Каталог FLUX 3 Image + flare: снапим неизвестные пропорции к ближайшему.
POLZA_ASPECTS: tuple[tuple[int, int], ...] = (
    (1, 1),
    (5, 4),
    (4, 5),
    (4, 3),
    (3, 4),
    (3, 2),
    (2, 3),
    (7, 5),
    (5, 7),
    (16, 9),
    (9, 16),
    (2, 1),
    (1, 2),
    (21, 9),
    (9, 21),
)
_FLUX_RESOLUTIONS = frozenset({"768", "1K", "1.5K", "2K", "4K"})
_PIXEL_SIZE = re.compile(r"^(\d+)\s*[xX×]\s*(\d+)$")
_ASPECT_RATIO = re.compile(r"^(\d+)\s*:\s*(\d+)$")
_TIER_SIZE = re.compile(r"^(\d+(?:\.\d+)?)\s*[Kk]$")

_PENDING_STATUSES = frozenset({"pending", "processing", "queued", "in_progress"})
_FAILED_STATUSES = frozenset({"failed", "cancelled", "canceled", "error"})

_STATUS_CODES: dict[int, str] = {
    400: "BAD_REQUEST",
    401: "UNAUTHORIZED",
    402: "INSUFFICIENT_BALANCE",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    408: "REQUEST_TIMEOUT",
    413: "PAYLOAD_TOO_LARGE",
    429: "TOO_MANY_REQUESTS",
    502: "BAD_GATEWAY",
    503: "SERVICE_UNAVAILABLE",
}

# Тексты для пользователя: без trace_id, апстрима и внутренних полей.
_UNAVAILABLE_TEXT = (
    "Сервис генерации изображений временно недоступен. Попробуйте позже."
)
_VIDEO_UNAVAILABLE_TEXT = (
    "Сервис генерации видео временно недоступен. Попробуйте позже."
)
_USER_MESSAGES: dict[str, str] = {
    "BAD_REQUEST": (
        "Не удалось обработать запрос на изображение. "
        "Измените описание или параметры и попробуйте снова."
    ),
    "UNAUTHORIZED": "Сервис генерации изображений сейчас недоступен (ошибка доступа).",
    "API_KEY_REVOKED": (
        "Сервис генерации изображений сейчас недоступен (ошибка доступа)."
    ),
    "FORBIDDEN": "Сервис генерации изображений сейчас недоступен (ошибка доступа).",
    "INSUFFICIENT_BALANCE": _UNAVAILABLE_TEXT,
    "NOT_FOUND": "Сервис генерации изображений не нашёл запрошенный ресурс.",
    "PAYLOAD_TOO_LARGE": "Изображение слишком большое. Отправьте файл поменьше.",
    "TOO_MANY_REQUESTS": (
        "Сервис генерации изображений перегружен. Попробуйте через минуту."
    ),
    "REQUEST_TIMEOUT": _UNAVAILABLE_TEXT,
    "BAD_GATEWAY": _UNAVAILABLE_TEXT,
    "SERVICE_UNAVAILABLE": _UNAVAILABLE_TEXT,
    "INTERNAL_ERROR": _UNAVAILABLE_TEXT,
    "TIMEOUT": "Превышено время ожидания сервиса генерации изображений.",
    "NETWORK_ERROR": "Сетевая ошибка при обращении к сервису генерации изображений.",
    "POLL_TIMEOUT": "Превышено время ожидания сервиса генерации изображений.",
    "UNOFFICIAL_PROVIDERS_DISABLED": (
        "Для этого запроса нет доступного маршрута генерации."
    ),
}
_VIDEO_USER_MESSAGES: dict[str, str] = {
    "BAD_REQUEST": (
        "Не удалось обработать запрос на видео. "
        "Измените описание или параметры и попробуйте снова."
    ),
    "UNAUTHORIZED": "Сервис генерации видео сейчас недоступен (ошибка доступа).",
    "API_KEY_REVOKED": "Сервис генерации видео сейчас недоступен (ошибка доступа).",
    "FORBIDDEN": "Сервис генерации видео сейчас недоступен (ошибка доступа).",
    "INSUFFICIENT_BALANCE": _VIDEO_UNAVAILABLE_TEXT,
    "NOT_FOUND": "Сервис генерации видео не нашёл запрошенный ресурс.",
    "PAYLOAD_TOO_LARGE": "Входные данные слишком большие для видео. Упростите запрос.",
    "TOO_MANY_REQUESTS": (
        "Сервис генерации видео перегружен. Попробуйте через минуту."
    ),
    "REQUEST_TIMEOUT": _VIDEO_UNAVAILABLE_TEXT,
    "BAD_GATEWAY": _VIDEO_UNAVAILABLE_TEXT,
    "SERVICE_UNAVAILABLE": _VIDEO_UNAVAILABLE_TEXT,
    "INTERNAL_ERROR": _VIDEO_UNAVAILABLE_TEXT,
    "TIMEOUT": "Превышено время ожидания сервиса генерации видео.",
    "NETWORK_ERROR": "Сетевая ошибка при обращении к сервису генерации видео.",
    "POLL_TIMEOUT": "Превышено время ожидания сервиса генерации видео.",
    "UNOFFICIAL_PROVIDERS_DISABLED": (
        "Для этого запроса нет доступного маршрута генерации видео."
    ),
}


async def _sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


@dataclass(frozen=True)
class MediaResult:
    """Результат ``/v1/media``: картинки, ``usage`` и предупреждения."""

    images: list[dict[str, str]]
    usage: dict[str, Any] | None = None
    warnings: list[Any] = field(default_factory=list)
    media_id: str | None = None
    status: str = "completed"

    @property
    def cost_rub(self) -> float | None:
        """Фактическая цена в рублях из ``usage`` (``cost_rub`` или алиас ``cost``)."""
        return extract_cost_rub(self.usage)

    def as_images_response(self) -> dict[str, Any]:
        """Нормализует в форму ``images/generations``: ``{data: [...], usage}``."""
        items: list[dict[str, str]] = []
        for record in self.images:
            if record.get("type") == "url":
                items.append({"url": record["value"]})
            else:
                items.append({"b64_json": record["value"]})
        response: dict[str, Any] = {"data": items}
        if self.usage is not None:
            usage = dict(self.usage)
            rub = self.cost_rub
            if rub is not None:
                # Алиас ``cost`` → ``cost_rub``: учёт читает одно поле.
                usage["cost_rub"] = rub
            response["usage"] = usage
        return response


def snap_aspect_ratio(width: int, height: int) -> str:
    """Ближайшее соотношение сторон из набора Polza."""
    if width <= 0 or height <= 0:
        return "1:1"
    target = width / height
    numer, denom = min(
        POLZA_ASPECTS,
        key=lambda item: abs((item[0] / item[1]) - target),
    )
    return f"{numer}:{denom}"


def image_resolution_tier(width: int, height: int) -> str:
    """Ярус ``image_resolution`` по длинной стороне (FLUX / каталог Polza).

    Для ``gpt-image-2.6-flare`` поле даёт 400 и **не отправляется**.
    """
    long_side = max(width, height)
    if long_side <= 768:
        return "768"
    if long_side <= 1280:
        return "1K"
    if long_side <= 1792:
        return "1.5K"
    if long_side <= 2560:
        return "2K"
    return "4K"


# ProxyAPI id → id в каталоге Polza ``/v1/models``.
_POLZA_MODEL_ALIASES: dict[str, str] = {
    "qwen/qwen-image-2.1-pro": "qwen/image-2.1",
    "qwen/qwen-image-2.1-turbo": "qwen/image-2.1",
}

_RESOLUTION_RANK: tuple[str, ...] = ("768", "1K", "1.5K", "2K", "4K")


@dataclass(frozen=True)
class PolzaMediaProfile:
    """Контракт ``/v1/media`` для семейства модели (из каталога Polza)."""

    model_id: str
    mode: str  # "openai_pixels" | "resolution"
    resolutions: frozenset[str]
    allow_quality: bool
    allow_output_format: bool
    pin_upstream: bool
    send_max_price: bool
    allow_aspect_auto: bool
    unsupported: bool = False


def resolve_polza_model_id(model: str) -> str:
    """Id модели для Polza: алиас ProxyAPI→каталог или исходный id."""
    cleaned = model.strip()
    return _POLZA_MODEL_ALIASES.get(cleaned.casefold(), cleaned)


def polza_media_profile(model: str) -> PolzaMediaProfile:
    """Профиль параметров Polza для модели из каталога бота."""
    resolved = resolve_polza_model_id(model)
    mid = resolved.casefold()
    if "recraft" in mid:
        return PolzaMediaProfile(
            model_id=resolved,
            mode="resolution",
            resolutions=frozenset(),
            allow_quality=False,
            allow_output_format=False,
            pin_upstream=False,
            send_max_price=False,
            allow_aspect_auto=False,
            unsupported=True,
        )
    if "nano-banana" in mid or "gemini-nano-banana" in mid:
        return PolzaMediaProfile(
            model_id=resolved,
            mode="resolution",
            resolutions=frozenset({"1K", "2K", "4K"}),
            allow_quality=False,
            allow_output_format=True,
            pin_upstream=False,
            send_max_price=False,
            allow_aspect_auto=True,
        )
    if mid.startswith("openai/") or "gpt-image" in mid:
        # Token pricing: ``max_price.image`` → 400 «цена не определена».
        return PolzaMediaProfile(
            model_id=resolved,
            mode="openai_pixels",
            resolutions=frozenset(),
            allow_quality=True,
            allow_output_format=True,
            pin_upstream=True,
            send_max_price=False,
            allow_aspect_auto=True,
        )
    if "klein" in mid:
        # ``unitParam=max_images``: max_price с image_resolution даёт 400.
        return PolzaMediaProfile(
            model_id=resolved,
            mode="resolution",
            resolutions=frozenset({"1K", "2K", "4K"}),
            allow_quality=False,
            allow_output_format=True,
            pin_upstream=False,
            send_max_price=False,
            allow_aspect_auto=False,
        )
    if "flux" in mid or "black-forest" in mid:
        return PolzaMediaProfile(
            model_id=resolved,
            mode="resolution",
            resolutions=frozenset({"768", "1K", "1.5K", "2K", "4K"}),
            allow_quality=False,
            allow_output_format=False,
            pin_upstream=False,
            send_max_price=True,
            allow_aspect_auto=True,
        )
    if mid.startswith("qwen/") or "qwen-image" in mid:
        return PolzaMediaProfile(
            model_id=resolved,
            mode="resolution",
            resolutions=frozenset({"1K", "2K"}),
            allow_quality=False,
            allow_output_format=True,
            pin_upstream=False,
            send_max_price=True,
            allow_aspect_auto=False,
        )
    return PolzaMediaProfile(
        model_id=resolved,
        mode="openai_pixels",
        resolutions=frozenset(),
        allow_quality=True,
        allow_output_format=True,
        pin_upstream=False,
        send_max_price=False,
        allow_aspect_auto=True,
    )


def uses_image_resolution(model: str) -> bool:
    """True, если каталог принимает ``image_resolution``, не pixel ``size``."""
    return polza_media_profile(model).mode == "resolution"


def _snap_resolution(raw: str, allowed: frozenset[str]) -> str:
    """Ближайший ярус из ``allowed`` (каталог модели)."""
    if not allowed:
        return raw
    if raw in allowed:
        return raw
    matched = next(
        (item for item in allowed if item.casefold() == raw.casefold()), None
    )
    if matched is not None:
        return matched
    rank = {name: index for index, name in enumerate(_RESOLUTION_RANK)}
    target = rank.get(raw, rank.get("1K", 1))
    # При ничьей берём больший ярус (1.5K → 2K, не 1K).
    return min(
        allowed,
        key=lambda item: (abs(rank.get(item, 99) - target), -rank.get(item, 0)),
    )


def polza_size_fields(size: str | None) -> dict[str, str]:
    """Размер из ``fit_image_size`` / аргумента → поля ``aspect_ratio`` и ``size``.

    Пиксели сохраняются в ``size`` (``WIDTHxHEIGHT``), ``aspect_ratio`` — ближайший
    из ``POLZA_ASPECTS``. ``auto`` → ``aspect_ratio: auto`` без размера.
    Для OpenAI-моделей на Polza; FLUX/Qwen — через ``polza_request_fields``.
    """
    cleaned = (size or "").strip()
    if not cleaned or cleaned.casefold() in {"auto", "adaptive"}:
        return {"aspect_ratio": "auto"}
    pixel = _PIXEL_SIZE.fullmatch(cleaned)
    if pixel is not None:
        width, height = int(pixel.group(1)), int(pixel.group(2))
        return {
            "aspect_ratio": snap_aspect_ratio(width, height),
            "size": f"{width}x{height}",
        }
    aspect = _ASPECT_RATIO.fullmatch(cleaned)
    if aspect is not None:
        return {
            "aspect_ratio": snap_aspect_ratio(
                int(aspect.group(1)) * 1000, int(aspect.group(2)) * 1000
            )
        }
    tier = _TIER_SIZE.fullmatch(cleaned)
    if tier is not None:
        return {"size": f"{tier.group(1)}K"}
    return {"size": cleaned}


def polza_request_fields(
    model: str,
    size: str | None,
    *,
    quality: str | None = None,
) -> dict[str, Any]:
    """Поля ``/v1/media`` + флаги роутинга под каталог модели.

    Возвращает также ``model`` (с алиасом Polza) и ``unsupported``.
    """
    profile = polza_media_profile(model)
    if profile.unsupported:
        return {
            "model": profile.model_id,
            "aspect_ratio": None,
            "size": None,
            "image_resolution": None,
            "quality": None,
            "output_format": None,
            "pin_upstream": False,
            "send_max_price": False,
            "unsupported": True,
        }

    if profile.mode == "openai_pixels":
        fields = polza_size_fields(size)
        return {
            "model": profile.model_id,
            **fields,
            "image_resolution": None,
            "quality": quality if profile.allow_quality else None,
            "output_format": "png" if profile.allow_output_format else None,
            "pin_upstream": profile.pin_upstream,
            "send_max_price": profile.send_max_price,
            "unsupported": False,
        }

    cleaned = (size or "").strip()
    aspect: str | None = None
    resolution: str | None = None
    if not cleaned or cleaned.casefold() in {"auto", "adaptive"}:
        aspect = "auto" if profile.allow_aspect_auto else "1:1"
    else:
        pixel = _PIXEL_SIZE.fullmatch(cleaned)
        if pixel is not None:
            width, height = int(pixel.group(1)), int(pixel.group(2))
            aspect = snap_aspect_ratio(width, height)
            resolution = _snap_resolution(
                image_resolution_tier(width, height), profile.resolutions
            )
        else:
            ratio = _ASPECT_RATIO.fullmatch(cleaned)
            if ratio is not None:
                aspect = snap_aspect_ratio(
                    int(ratio.group(1)) * 1000, int(ratio.group(2)) * 1000
                )
            else:
                tier = _TIER_SIZE.fullmatch(cleaned)
                if cleaned in _FLUX_RESOLUTIONS or cleaned in profile.resolutions:
                    candidate = cleaned
                elif tier is not None:
                    candidate = f"{tier.group(1)}K"
                else:
                    candidate = cleaned
                resolution = _snap_resolution(candidate, profile.resolutions)

    return {
        "model": profile.model_id,
        "aspect_ratio": aspect,
        "size": None,
        "image_resolution": resolution,
        "quality": None,
        "output_format": "png" if profile.allow_output_format else None,
        "pin_upstream": profile.pin_upstream,
        "send_max_price": profile.send_max_price,
        "unsupported": False,
    }


def hash_user_id(user_id: int | str | None) -> str | None:
    """Хеш telegram id для поля ``user`` (сырой id наружу не уходит).

    With ``POLZA_USER_HASH_SECRET`` uses HMAC-SHA256 so the digest is not
    dictionary-guessable from a known Telegram id.
    """
    if user_id is None:
        return None
    secret = (os.getenv("POLZA_USER_HASH_SECRET") or "").strip()
    material = f"assist-ai:{user_id}".encode()
    if secret:
        digest = hmac.new(secret.encode(), material, hashlib.sha256).hexdigest()
    else:
        digest = hashlib.sha256(material).hexdigest()
    return digest[:_MAX_USER_LEN]


def build_media_body(
    provider: ProviderConfig,
    *,
    model: str,
    prompt: str,
    aspect_ratio: str | None = None,
    size: str | None = None,
    image_resolution: str | None = None,
    quality: str | None = None,
    output_format: str | None = "png",
    images: list[bytes] | None = None,
    max_images: int | None = 1,
    user: str | None = None,
    pin_upstream: bool | None = None,
    send_max_price: bool | None = None,
) -> dict[str, Any]:
    """Тело ``POST /v1/media``. ``complexity`` и ``background`` не включаются."""
    media_input: dict[str, Any] = {"prompt": prompt}
    if aspect_ratio:
        media_input["aspect_ratio"] = aspect_ratio
    if size:
        media_input["size"] = size
    if image_resolution:
        media_input["image_resolution"] = image_resolution
    if quality:
        media_input["quality"] = quality
    if output_format:
        media_input["output_format"] = output_format
    if max_images:
        media_input["max_images"] = max_images
    if images:
        # Порядок сохраняется: основа — первой, как в prepare_edit_files.
        media_input["images"] = [
            {"type": "base64", "data": base64.b64encode(raw).decode("ascii")}
            for raw in images
        ]
    body: dict[str, Any] = {"model": model, "input": media_input}
    if user:
        body["user"] = user[:_MAX_USER_LEN]
    profile = polza_media_profile(model)
    pin = profile.pin_upstream if pin_upstream is None else pin_upstream
    price = profile.send_max_price if send_max_price is None else send_max_price
    routing = provider.media_provider_object(pin_upstream=pin, send_max_price=price)
    if routing is not None:
        body["provider"] = routing
    return body


def _parse_retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if raw is None:
        return None
    try:
        value = float(raw.strip())
    except ValueError:
        return None
    return value if value >= 0 else None


def _error_parts(body: Any) -> tuple[str | None, str | None, dict[str, Any]]:
    """(code, trace_id, metadata) из тела ошибки Polza."""
    if not isinstance(body, dict):
        return None, None, {}
    error = body.get("error")
    trace_id = body.get("trace_id")
    if not isinstance(error, dict):
        return None, trace_id if isinstance(trace_id, str) else None, {}
    code = error.get("code")
    nested_trace = error.get("trace_id")
    metadata = error.get("metadata")
    return (
        str(code) if code else None,
        (
            str(nested_trace)
            if nested_trace
            else (trace_id if isinstance(trace_id, str) else None)
        ),
        metadata if isinstance(metadata, dict) else {},
    )


def _user_message(code: str, *, medium: str = "image") -> str:
    upper = code.upper()
    if upper.startswith("GUARD_"):
        if medium == "video":
            return "Запрос отклонён политикой безопасности сервиса генерации видео."
        return "Запрос отклонён политикой безопасности сервиса генерации."
    if medium == "video":
        return _VIDEO_USER_MESSAGES.get(upper, _VIDEO_UNAVAILABLE_TEXT)
    return _USER_MESSAGES.get(upper, _UNAVAILABLE_TEXT)


def map_media_error(
    body: Any,
    *,
    status_code: int | None,
    retry_after: float | None = None,
    provider_id: str = "polza",
    medium: str = "image",
    job_committed: bool = False,
) -> AssistAIError:
    """``error.code`` → ``AssistAIError``. Внутренности только в лог."""
    code, trace_id, metadata = _error_parts(body)
    if code is None:
        if status_code is not None and status_code in _STATUS_CODES:
            code = _STATUS_CODES[status_code]
        elif status_code is not None and status_code >= 500:
            code = "INTERNAL_ERROR"
        else:
            code = "BAD_REQUEST" if status_code and status_code >= 400 else "FAILED"
    logger.error(
        "polza_media_error",
        provider=provider_id,
        code=code,
        status_code=status_code,
        trace_id=trace_id,
        reason=metadata.get("reason"),
        provider_name=metadata.get("provider_name"),
        medium=medium,
    )
    return AssistAIError(
        _user_message(code, medium=medium),
        status_code=status_code,
        error_code=code,
        retry_after=retry_after,
        job_committed=job_committed,
    )


def _raise_for_response(
    response: httpx.Response,
    provider_id: str,
    *,
    medium: str = "image",
) -> None:
    if response.status_code < 400:
        return
    try:
        body: Any = response.json()
    except ValueError:
        body = None
    raise map_media_error(
        body,
        status_code=response.status_code,
        retry_after=_parse_retry_after(response),
        provider_id=provider_id,
        medium=medium,
    )


def _remember_polza_job(exc: AssistAIError, media_id: str, *, medium: str) -> None:
    """Уже принятая задача: бот заберёт файл позже, без второй оплаты."""
    exc.job_committed = True
    exc.pending_provider = "polza"
    exc.pending_job_id = media_id
    exc.pending_kind = "video" if medium == "video" else "image"


def _items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    data = payload.get("data")
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


def parse_media_payload(payload: dict[str, Any]) -> MediaResult:
    """Разбор completed-ответа: ``data`` как список или объект, url или b64_json."""
    records: list[dict[str, str]] = []
    for item in _items(payload):
        url = item.get("url")
        encoded = item.get("b64_json")
        if isinstance(url, str) and url:
            records.append({"type": "url", "value": url})
        elif isinstance(encoded, str) and encoded:
            records.append({"type": "base64", "value": encoded})
    usage = payload.get("usage")
    warnings = payload.get("warnings")
    media_id = payload.get("id")
    return MediaResult(
        images=records,
        usage=usage if isinstance(usage, dict) else None,
        warnings=warnings if isinstance(warnings, list) else [],
        media_id=media_id if isinstance(media_id, str) else None,
        status=str(payload.get("status") or "completed"),
    )


def _log_warnings(result: MediaResult) -> None:
    if result.warnings:
        logger.warning(
            "polza_media_warnings",
            media_id=result.media_id,
            warnings=result.warnings,
        )


def _network_error(
    exc: httpx.RequestError,
    operation: str,
    *,
    medium: str = "image",
) -> AssistAIError:
    messages = _VIDEO_USER_MESSAGES if medium == "video" else _USER_MESSAGES
    if isinstance(exc, httpx.TimeoutException):
        logger.error("polza_media_timeout", operation=operation, medium=medium)
        return AssistAIError(messages["TIMEOUT"], error_code="TIMEOUT")
    logger.error(
        "polza_media_network",
        operation=operation,
        medium=medium,
        error=type(exc).__name__,
    )
    return AssistAIError(messages["NETWORK_ERROR"], error_code="NETWORK_ERROR")


def _capped_timeout(base: httpx.Timeout | None, remaining: float) -> httpx.Timeout:
    """HTTP timeout that never exceeds the poll wall-clock budget."""
    budget = max(0.05, remaining)
    connect = min(10.0, budget)
    read = budget
    if base is not None:
        if base.connect is not None:
            connect = min(float(base.connect), budget)
        if base.read is not None:
            read = min(float(base.read), budget)
    return httpx.Timeout(read, connect=connect)


async def _poll(
    client: httpx.AsyncClient,
    provider: ProviderConfig,
    media_id: str,
    *,
    poll_interval: float,
    total_wait: float,
    timeout: httpx.Timeout | None,
    medium: str = "image",
) -> dict[str, Any]:
    """Опрос ``GET /v1/media/{id}`` до терминального статуса или ``total_wait``.

    ``total_wait`` is a hard budget: poll sleeps accumulate toward it (so mocked
    sleeps cannot busy-loop), and each GET is also capped by the wall-clock
    remainder so a long HTTP timeout cannot overrun the deadline.
    """
    messages = _VIDEO_USER_MESSAGES if medium == "video" else _USER_MESSAGES
    unavailable = _VIDEO_UNAVAILABLE_TEXT if medium == "video" else _UNAVAILABLE_TEXT
    url = f"{provider.base_url}{_MEDIA_PATH}/{quote(media_id, safe='')}"
    interval = max(0.0, poll_interval)
    budget = max(0.0, total_wait)
    deadline = time.monotonic() + budget
    slept = 0.0
    errors = 0
    while True:
        remaining_wall = deadline - time.monotonic()
        remaining_sleep = budget - slept
        if remaining_sleep <= 0 or remaining_wall <= 0:
            logger.error(
                "polza_media_poll_exhausted",
                media_id=media_id,
                waited=total_wait,
                medium=medium,
            )
            raise AssistAIError(
                messages["POLL_TIMEOUT"],
                error_code="POLL_TIMEOUT",
                job_committed=True,
                pending_provider="polza",
                pending_job_id=media_id,
                pending_kind="video" if medium == "video" else "image",
            )
        sleep_for = (
            min(interval, remaining_sleep, remaining_wall) if interval > 0 else 0.0
        )
        if sleep_for > 0:
            await _sleep(sleep_for)
            slept += sleep_for
        remaining_wall = max(0.0, deadline - time.monotonic())
        # One GET after each wait slice; cap HTTP to leftover wall (floor 50ms).
        try:
            response = await client.get(
                url, timeout=_capped_timeout(timeout, max(remaining_wall, 0.05))
            )
            _raise_for_response(response, provider.id, medium=medium)
            body = response.json()
        except httpx.RequestError as exc:
            errors += 1
            # Видео уже оплачено: сетевой сбой не бросает задачу до конца ожидания.
            if medium != "video" and errors > _MAX_POLL_ERRORS:
                raise _network_error(exc, "poll", medium=medium) from exc
            continue
        except AssistAIError as exc:
            if exc.status_code is not None and exc.status_code >= 500:
                errors += 1
                if medium == "video" or errors <= _MAX_POLL_ERRORS:
                    continue
            raise
        except ValueError as exc:
            raise AssistAIError(unavailable, error_code="BAD_GATEWAY") from exc
        if not isinstance(body, dict):
            raise AssistAIError(unavailable, error_code="BAD_GATEWAY")
        errors = 0
        status = str(body.get("status") or "").casefold()
        if status in _PENDING_STATUSES:
            continue
        return body


async def generate(
    client: httpx.AsyncClient,
    provider: ProviderConfig,
    *,
    model: str,
    prompt: str,
    aspect_ratio: str | None = None,
    size: str | None = None,
    image_resolution: str | None = None,
    quality: str | None = None,
    output_format: str | None = "png",
    images: list[bytes] | None = None,
    max_images: int | None = 1,
    user: str | None = None,
    pin_upstream: bool | None = None,
    send_max_price: bool | None = None,
    poll_interval: float = 4.0,
    total_wait: float = 240.0,
    timeout: httpx.Timeout | None = None,
) -> MediaResult:
    """Генерация (без ``images``) или правка (с ``images``) через ``/v1/media``.

    Если вместо ``data`` пришли ``id`` + ``pending|processing``, опрашивает статус
    раз в ``poll_interval`` секунд до ``total_wait`` и при исчерпании поднимает
    ``AssistAIError(error_code="POLL_TIMEOUT")``.
    """
    if images is not None and len(images) > POLZA_MAX_REFERENCE_IMAGES:
        raise AssistAIError(
            f"Слишком много изображений: максимум {POLZA_MAX_REFERENCE_IMAGES}.",
            error_code="BAD_REQUEST",
        )
    body = build_media_body(
        provider,
        model=model,
        prompt=prompt,
        aspect_ratio=aspect_ratio,
        size=size,
        image_resolution=image_resolution,
        quality=quality,
        output_format=output_format,
        images=images,
        max_images=max_images,
        user=user,
        pin_upstream=pin_upstream,
        send_max_price=send_max_price,
    )
    url = f"{provider.base_url}{_MEDIA_PATH}"
    logger.info(
        "polza_media_request",
        model=model,
        aspect_ratio=aspect_ratio,
        size=size,
        image_resolution=image_resolution,
        quality=quality,
        refs=len(images or []),
        prompt_chars=len(prompt),
    )
    try:
        response = await client.post(url, json=body, timeout=timeout)
    except httpx.RequestError as exc:
        raise _network_error(exc, "create") from exc
    _raise_for_response(response, provider.id)
    # Successful create accepted by Polza — further poll/parse failures must not
    # trigger a second paid job on the fallback provider (same as generate_video).
    try:
        try:
            payload = response.json()
        except ValueError as exc:
            raise AssistAIError(
                _UNAVAILABLE_TEXT, error_code="BAD_GATEWAY", job_committed=True
            ) from exc
        if not isinstance(payload, dict):
            raise AssistAIError(
                _UNAVAILABLE_TEXT, error_code="BAD_GATEWAY", job_committed=True
            )

        status = str(payload.get("status") or "").casefold()
        media_id = payload.get("id")
        if (
            status in _PENDING_STATUSES
            and not _items(payload)
            and isinstance(media_id, str)
            and media_id
        ):
            logger.info("polza_media_async", media_id=media_id, status=status)
            try:
                payload = await _poll(
                    client,
                    provider,
                    media_id,
                    poll_interval=poll_interval,
                    total_wait=total_wait,
                    timeout=timeout,
                )
            except AssistAIError as exc:
                _remember_polza_job(exc, media_id, medium="image")
                raise
            status = str(payload.get("status") or "").casefold()

        if status in _FAILED_STATUSES or (payload.get("error") and not _items(payload)):
            raise map_media_error(
                payload,
                status_code=None,
                provider_id=provider.id,
                job_committed=True,
            )

        result = parse_media_payload(payload)
        _log_warnings(result)
        if not result.images:
            logger.error(
                "polza_media_no_image", media_id=result.media_id, status=status
            )
            raise AssistAIError(
                "Сервис генерации не вернул изображение.",
                error_code="SERVICE_UNAVAILABLE",
                job_committed=True,
            )
    except AssistAIError as exc:
        exc.job_committed = True
        raise
    logger.info(
        "polza_media_done",
        media_id=result.media_id,
        images=len(result.images),
        cost_rub=result.cost_rub,
    )
    return result


def build_video_body(
    *,
    model: str,
    prompt: str,
    duration: int | None = None,
    resolution: str | None = None,
    aspect_ratio: str = "16:9",
    images: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Тело ``POST /v1/media`` для Seedance и др. видеомоделей Polza."""
    media_input: dict[str, Any] = {
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
    }
    if isinstance(duration, int) and duration > 0:
        media_input["duration"] = str(duration)
    if isinstance(resolution, str) and resolution.strip():
        media_input["resolution"] = resolution.strip()
    if images:
        media_input["images"] = images
    return {"model": model, "input": media_input, "async": True}


async def generate_video(
    client: httpx.AsyncClient,
    provider: ProviderConfig,
    *,
    model: str,
    prompt: str,
    duration: int | None = None,
    resolution: str | None = None,
    aspect_ratio: str = "16:9",
    images: list[dict[str, str]] | None = None,
    poll_interval: float = 5.0,
    total_wait: float = 600.0,
    timeout: httpx.Timeout | None = None,
) -> MediaResult:
    """Асинхронная генерация видео через Polza ``/v1/media`` + poll."""
    body = build_video_body(
        model=model,
        prompt=prompt,
        duration=duration,
        resolution=resolution,
        aspect_ratio=aspect_ratio,
        images=images,
    )
    url = f"{provider.base_url}{_MEDIA_PATH}"
    logger.info(
        "polza_video_request",
        model=model,
        duration=duration,
        resolution=resolution,
        prompt_chars=len(prompt),
        images=len(images or []),
    )
    try:
        response = await client.post(url, json=body, timeout=timeout)
    except httpx.RequestError as exc:
        raise _network_error(exc, "video_create", medium="video") from exc
    _raise_for_response(response, provider.id, medium="video")
    # Successful create accepted by Polza — further poll/parse failures must not
    # trigger a second paid job on the fallback provider.
    try:
        try:
            payload = response.json()
        except ValueError as exc:
            raise AssistAIError(
                _VIDEO_UNAVAILABLE_TEXT,
                error_code="BAD_GATEWAY",
                job_committed=True,
            ) from exc
        if not isinstance(payload, dict):
            raise AssistAIError(
                _VIDEO_UNAVAILABLE_TEXT,
                error_code="BAD_GATEWAY",
                job_committed=True,
            )

        status = str(payload.get("status") or "").casefold()
        media_id = payload.get("id")
        if (
            status in _PENDING_STATUSES
            and not _items(payload)
            and isinstance(media_id, str)
            and media_id
        ):
            logger.info("polza_video_async", media_id=media_id, status=status)
            try:
                payload = await _poll(
                    client,
                    provider,
                    media_id,
                    poll_interval=poll_interval,
                    total_wait=total_wait,
                    timeout=timeout,
                    medium="video",
                )
            except AssistAIError as exc:
                _remember_polza_job(exc, media_id, medium="video")
                raise
            status = str(payload.get("status") or "").casefold()

        if status in _FAILED_STATUSES or (payload.get("error") and not _items(payload)):
            raise map_media_error(
                payload,
                status_code=None,
                provider_id=provider.id,
                medium="video",
                job_committed=True,
            )

        result = parse_media_payload(payload)
        _log_warnings(result)
        if not result.images:
            logger.error("polza_video_no_file", media_id=result.media_id, status=status)
            raise AssistAIError(
                "Сервис генерации не вернул видео.",
                error_code="SERVICE_UNAVAILABLE",
                job_committed=True,
            )
    except AssistAIError as exc:
        exc.job_committed = True
        raise
    logger.info(
        "polza_video_done",
        media_id=result.media_id,
        files=len(result.images),
        cost_rub=result.cost_rub,
    )
    return result
