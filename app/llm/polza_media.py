"""Адаптер Polza.ai Media API: генерация и правка картинок через ``POST /v1/media``.

Контракт проверен живым прогоном (``app/docs/polza-media-contract.md``):

* для ``openai/gpt-image-2.6-flare`` ``image_resolution`` даёт HTTP 400,
  размер задаётся ``aspect_ratio`` + пиксельным ``size`` (``1536x1024``);
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
import re
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

import httpx
import structlog

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
POLZA_ASPECTS: tuple[tuple[int, int], ...] = (
    (1, 1),
    (5, 4),
    (4, 5),
    (4, 3),
    (3, 4),
    (3, 2),
    (2, 3),
    (16, 9),
    (9, 16),
    (21, 9),
    (9, 21),
)
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
        if not isinstance(self.usage, dict):
            return None
        for key in ("cost_rub", "cost"):
            value = self.usage.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value) if value >= 0 else None
        return None

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
    """Ярус ``image_resolution`` по длинной стороне (для моделей, где он поддержан).

    Для ``gpt-image-2.6-flare`` поле даёт 400 и **не отправляется**.
    """
    long_side = max(width, height)
    if long_side <= 1536:
        return "1K"
    if long_side <= 2560:
        return "2K"
    return "4K"


def polza_size_fields(size: str | None) -> dict[str, str]:
    """Размер из ``fit_image_size`` / аргумента → поля ``aspect_ratio`` и ``size``.

    Пиксели сохраняются в ``size`` (``WIDTHxHEIGHT``), ``aspect_ratio`` — ближайший
    из ``POLZA_ASPECTS``. ``auto`` → ``aspect_ratio: auto`` без размера.
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


def hash_user_id(user_id: int | str | None) -> str | None:
    """Хеш telegram id для поля ``user`` (сырой id наружу не уходит)."""
    if user_id is None:
        return None
    digest = hashlib.sha256(f"assist-ai:{user_id}".encode()).hexdigest()
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
    routing = provider.media_provider_object()
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


def _user_message(code: str) -> str:
    upper = code.upper()
    if upper.startswith("GUARD_"):
        return "Запрос отклонён политикой безопасности сервиса генерации."
    return _USER_MESSAGES.get(upper, _UNAVAILABLE_TEXT)


def map_media_error(
    body: Any,
    *,
    status_code: int | None,
    retry_after: float | None = None,
    provider_id: str = "polza",
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
    )
    return AssistAIError(
        _user_message(code),
        status_code=status_code,
        error_code=code,
        retry_after=retry_after,
    )


def _raise_for_response(response: httpx.Response, provider_id: str) -> None:
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
    )


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


def _network_error(exc: httpx.RequestError, operation: str) -> AssistAIError:
    if isinstance(exc, httpx.TimeoutException):
        logger.error("polza_media_timeout", operation=operation)
        return AssistAIError(_USER_MESSAGES["TIMEOUT"], error_code="TIMEOUT")
    logger.error("polza_media_network", operation=operation, error=type(exc).__name__)
    return AssistAIError(_USER_MESSAGES["NETWORK_ERROR"], error_code="NETWORK_ERROR")


async def _poll(
    client: httpx.AsyncClient,
    provider: ProviderConfig,
    media_id: str,
    *,
    poll_interval: float,
    total_wait: float,
    timeout: httpx.Timeout | None,
) -> dict[str, Any]:
    """Опрос ``GET /v1/media/{id}`` до терминального статуса или ``total_wait``."""
    url = f"{provider.base_url}{_MEDIA_PATH}/{quote(media_id, safe='')}"
    interval = max(0.0, poll_interval)
    started = time.monotonic()
    waited = 0.0
    errors = 0
    while True:
        if max(time.monotonic() - started, waited) >= total_wait:
            logger.error(
                "polza_media_poll_exhausted", media_id=media_id, waited=total_wait
            )
            raise AssistAIError(
                _USER_MESSAGES["POLL_TIMEOUT"], error_code="POLL_TIMEOUT"
            )
        await _sleep(interval)
        waited += interval
        try:
            response = await client.get(url, timeout=timeout)
            _raise_for_response(response, provider.id)
            body = response.json()
        except httpx.RequestError as exc:
            errors += 1
            if errors > _MAX_POLL_ERRORS:
                raise _network_error(exc, "poll") from exc
            continue
        except AssistAIError as exc:
            if exc.status_code is not None and exc.status_code >= 500:
                errors += 1
                if errors <= _MAX_POLL_ERRORS:
                    continue
            raise
        except ValueError as exc:
            raise AssistAIError(_UNAVAILABLE_TEXT, error_code="BAD_GATEWAY") from exc
        if not isinstance(body, dict):
            raise AssistAIError(_UNAVAILABLE_TEXT, error_code="BAD_GATEWAY")
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
    )
    url = f"{provider.base_url}{_MEDIA_PATH}"
    logger.info(
        "polza_media_request",
        model=model,
        aspect_ratio=aspect_ratio,
        size=size,
        quality=quality,
        refs=len(images or []),
        prompt_chars=len(prompt),
    )
    try:
        response = await client.post(url, json=body, timeout=timeout)
    except httpx.RequestError as exc:
        raise _network_error(exc, "create") from exc
    _raise_for_response(response, provider.id)
    try:
        payload = response.json()
    except ValueError as exc:
        raise AssistAIError(_UNAVAILABLE_TEXT, error_code="BAD_GATEWAY") from exc
    if not isinstance(payload, dict):
        raise AssistAIError(_UNAVAILABLE_TEXT, error_code="BAD_GATEWAY")

    status = str(payload.get("status") or "").casefold()
    media_id = payload.get("id")
    if (
        status in _PENDING_STATUSES
        and not _items(payload)
        and isinstance(media_id, str)
        and media_id
    ):
        logger.info("polza_media_async", media_id=media_id, status=status)
        payload = await _poll(
            client,
            provider,
            media_id,
            poll_interval=poll_interval,
            total_wait=total_wait,
            timeout=timeout,
        )
        status = str(payload.get("status") or "").casefold()

    if status in _FAILED_STATUSES or (payload.get("error") and not _items(payload)):
        raise map_media_error(
            payload,
            status_code=None,
            provider_id=provider.id,
        )

    result = parse_media_payload(payload)
    _log_warnings(result)
    if not result.images:
        logger.error("polza_media_no_image", media_id=result.media_id, status=status)
        raise AssistAIError("Сервис генерации не вернул изображение.")
    logger.info(
        "polza_media_done",
        media_id=result.media_id,
        images=len(result.images),
        cost_rub=result.cost_rub,
    )
    return result
