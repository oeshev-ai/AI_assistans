"""Генерация и редактирование изображений, ImageModelScope, sizing."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Awaitable, Callable
from contextvars import ContextVar, Token
from typing import TYPE_CHECKING, Any, Literal, Self

import httpx
import structlog

from app.infra.circuit_breaker import CircuitBreakerError, get_breaker
from app.infra.cost_tracker import BudgetExceededError, cost_tracker, get_billing_user
from app.infra.metrics import note_provider_fallback
from app.infra.security import security
from app.llm import polza_media
from app.llm.errors import (
    AssistAIError,
    counts_toward_circuit,
    fallback_reason,
    is_rate_limited,
)
from app.llm.providers import ProviderConfig
from app.llm.tooling import BaseTool, _error_json, _exception_text
from app.llm.transport import (
    _IMAGE_READ_TIMEOUT,
    _IMAGES_EDIT_PATH,
    _IMAGES_PATH,
    _MAX_RETRY_AFTER_SECONDS,
    _RATE_LIMIT_RETRIES,
    ProviderTransportMixin,
    _provider_label,
    _response_error_text,
    _retry_after_seconds,
    _sleep,
    _ssl_context,
)

logger = structlog.get_logger("AssistAI.image")

if TYPE_CHECKING:
    from app.llm.assist_ai import AssistAIClient

_REQUEST_IMAGE_MODEL: ContextVar[str | None] = ContextVar(
    "request_image_model", default=None
)
_REQUEST_IMAGE_SIZE: ContextVar[str | None] = ContextVar(
    "request_image_size", default=None
)
_REQUEST_IMAGES: ContextVar[list[dict[str, str]] | None] = ContextVar(
    "request_generated_images", default=None
)
_GENERATION_IDS: ContextVar[list[str | None] | None] = ContextVar(
    "assist_ai_generation_ids", default=None
)


def _set_generation_ids(ids: list[str | None]) -> None:
    """Store image generation ids for the current asyncio task only."""
    _GENERATION_IDS.set(list(ids))


def _get_generation_ids() -> list[str | None]:
    value = _GENERATION_IDS.get()
    return list(value) if value else []


# FLUX принимает ярус разрешения, а не пиксели вроде 1024x1024.
_FLUX_SIZE_TIERS: tuple[tuple[int, str], ...] = (
    (768, "768"),
    (1024, "1K"),
    (1536, "1.5K"),
    (2048, "2K"),
    (4096, "4K"),
)
_FLUX_TIER_NAMES: dict[str, str] = {name.upper(): name for _, name in _FLUX_SIZE_TIERS}
_FLUX_ASPECTS: tuple[tuple[int, int], ...] = (
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
_PIXEL_SIZE = re.compile(r"(\d+)\s*[xX×]\s*(\d+)")
_ASPECT_RATIO = re.compile(r"^(\d+)\s*:\s*(\d+)$")
ImageModelFamily = Literal["openai", "flux", "qwen", "seedream", "recraft"]


class ImageModelScope:
    """На время запроса подставляет image-модель персоны и собирает картинки.

    Если модель сама передала ``model`` или ``size``, её значение важнее.
    Пустая строка означает дефолт клиента. Размер без явного ``size``
    берётся из персоны, а если его нет — ``1024x1024``.
    Байты и URL из ``generate_image`` копятся в ``images`` и не уходят
    обратно в контекст текстовой модели.
    """

    def __init__(self, model: str | None, size: str | None = None) -> None:
        self._model = (
            model.strip() if isinstance(model, str) and model.strip() else None
        )
        self._size = size.strip() if isinstance(size, str) and size.strip() else None
        self.images: list[dict[str, str]] = []
        self._token: Token[str | None] | None = None
        self._size_token: Token[str | None] | None = None
        self._images_token: Token[list[dict[str, str]] | None] | None = None

    def __enter__(self) -> Self:
        self._token = _REQUEST_IMAGE_MODEL.set(self._model)
        self._size_token = _REQUEST_IMAGE_SIZE.set(self._size)
        self._images_token = _REQUEST_IMAGES.set(self.images)
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._images_token is not None:
            _REQUEST_IMAGES.reset(self._images_token)
        if self._size_token is not None:
            _REQUEST_IMAGE_SIZE.reset(self._size_token)
        if self._token is not None:
            _REQUEST_IMAGE_MODEL.reset(self._token)


class GenerateImageTool(BaseTool):
    """Генерация изображения через ``AssistAIClient.generate_image``.

    Модель вызывает инструмент, когда пользователь просит картинку.
    Клиент передаётся снаружи: HTTP и выбор модели по умолчанию остаются в нём.
    """

    def __init__(self, client: AssistAIClient) -> None:
        self._client = client

    @property
    def name(self) -> str:
        return "generate_image"

    @property
    def description(self) -> str:
        return (
            "Сгенерировать изображение по текстовому описанию. "
            "Вызывай, когда пользователь просит нарисовать, сгенерировать или "
            "создать изображение, картинку, схему или иллюстрацию. "
            "Не отвечай, что не умеешь рисовать: используй этот инструмент. "
            "Если модель генерации не указана, берётся модель по умолчанию клиента."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "Детальное описание изображения на русском или английском."
                    ),
                },
                "model": {
                    "type": "string",
                    "description": (
                        "Модель генерации, например openai/gpt-image-2.6-flare. "
                        "Необязательно."
                    ),
                },
                "size": {
                    "type": "string",
                    "description": (
                        "Размер. OpenAI — пиксели (1024x1024). "
                        "FLUX/Seedream — ярус 1K или 2K (пиксели тоже принимаются). "
                        "Qwen/Recraft — соотношение сторон 1:1, 16:9, 9:16. "
                        "По умолчанию 1024x1024."
                    ),
                },
            },
            "required": ["prompt"],
            "additionalProperties": False,
        }

    async def execute(self, **kwargs: Any) -> str:
        """Генерирует картинку и возвращает короткий JSON для текстовой модели.

        Сами URL и base64 складываются в ``ImageModelScope`` и в ответ модели
        не попадают: иначе следующий запрос превышает лимит контекста.
        Ошибки аргументов и ProxyAPI становятся ``{"error": "..."}``.
        """
        prompt = kwargs.get("prompt")
        model = kwargs.get("model")
        raw_size = kwargs.get("size")
        if not isinstance(prompt, str) or not prompt.strip():
            return _error_json(
                "Параметр prompt обязателен и должен быть непустой строкой."
            )
        if model is not None and not isinstance(model, str):
            return _error_json("Параметр model должен быть строкой.")
        if raw_size is None:
            scoped = _REQUEST_IMAGE_SIZE.get()
            size = scoped if isinstance(scoped, str) and scoped.strip() else "1024x1024"
        elif not isinstance(raw_size, str) or not raw_size.strip():
            return _error_json("Параметр size должен быть строкой, например 1024x1024.")
        else:
            size = raw_size
        selected_model = (
            model.strip() if isinstance(model, str) and model.strip() else None
        )
        if selected_model is None:
            selected_model = _REQUEST_IMAGE_MODEL.get()
        try:
            images = await self._client.generate_image(
                prompt=prompt.strip(),
                model=selected_model,
                size=size.strip(),
            )
        except BudgetExceededError as exc:
            logger.warning("generate_image_budget_exceeded", error=exc.message)
            return _error_json(exc.message)
        except AssistAIError as exc:
            logger.error("generate_image: %s", exc.message)
            return _error_json(exc.message)
        except (TypeError, ValueError) as exc:
            logger.error("generate_image: %s", exc)
            return _error_json(_exception_text(exc))
        except Exception as exc:
            logger.exception("generate_image: непредвиденная ошибка")
            return _error_json(_exception_text(exc))

        generation_ids = _get_generation_ids()
        packed = []
        for index, image in enumerate(images):
            item = {
                "type": "url"
                if image.startswith(("http://", "https://"))
                else "base64",
                "value": image,
            }
            if index < len(generation_ids):
                generation_id = generation_ids[index]
                if isinstance(generation_id, str) and generation_id:
                    item["generation_id"] = generation_id
            packed.append(item)
        bucket = _REQUEST_IMAGES.get()
        if bucket is None:
            logger.warning(
                "generate_image: нет ImageModelScope, картинки некому отдать в чат"
            )
        else:
            bucket.extend(packed)
        logger.info("generate_image: получено изображений=%s", len(packed))
        return json.dumps(
            {
                "status": "success",
                "image_count": len(packed),
                "images": [{"type": item["type"]} for item in packed],
                "prompt": prompt.strip(),
                "note": (
                    "Изображение уже передаётся пользователю в Telegram. "
                    "Не вставляй base64 и не вызывай инструмент повторно. "
                    "Кратко подтверди, что картинка готова."
                ),
            },
            ensure_ascii=False,
        )


def note_generation_id(generation_id: str | None) -> None:
    """Кладёт id в scope текущего хода. В JSON для модели его не добавляют."""
    if not isinstance(generation_id, str) or not generation_id:
        return
    bucket = _REQUEST_IMAGES.get()
    if bucket is None:
        return
    bucket.append({"generation_id": generation_id})


def image_model_family(model: str) -> ImageModelFamily:
    """Семейство image-модели: от него зависит формат размера в запросе."""
    lowered = model.casefold()
    if "flux" in lowered or "black-forest" in lowered:
        return "flux"
    if "qwen-image" in lowered:
        return "qwen"
    if "seedream" in lowered or "bytedance-seed" in lowered:
        return "seedream"
    if "recraft" in lowered:
        return "recraft"
    return "openai"


def supports_multi_image(model_id: str, provider: ProviderConfig | None = None) -> bool:
    """True, если edits принимает несколько файлов.

    По карточкам каталога несколько ``image`` принимает семейство OpenAI
    (``openai/gpt-image-*``). Остальные семейства и неизвестные id — одно фото.
    На Polza (``/v1/media``) контракт проверен только для этого же семейства;
    лимит ``images`` — ``POLZA_MAX_REFERENCE_IMAGES`` (см. ``max_edit_images``).
    ``provider=None`` и ProxyAPI — прежнее поведение.
    """
    cleaned = model_id.strip()
    if not cleaned.casefold().startswith("openai/"):
        return False
    return image_model_family(cleaned) == "openai"


def max_edit_images(provider: ProviderConfig | None = None) -> int | None:
    """Лимит референсов у провайдера. ``None`` — без явного лимита (ProxyAPI)."""
    if provider is not None and provider.kind == "polza_media":
        return polza_media.POLZA_MAX_REFERENCE_IMAGES
    return None


def prepare_edit_files(
    files: list[tuple[bytes, str, str]],
    model_id: str,
    labels: list[str] | None = None,
    *,
    provider: ProviderConfig | None = None,
) -> list[tuple[bytes, str, str]]:
    """Оставляет файлы по отдельности или собирает одно полотно.

    OpenAI получает исходный порядок (основа первой). Модель без multi-image
    при ``ALLOW_COMPOSITE_FALLBACK=1`` склеивает кадры рядом, иначе ошибка.
    Референс при этом не отбрасывается молча.
    """
    if len(files) <= 1 or supports_multi_image(model_id, provider):
        limit = max_edit_images(provider)
        if limit is not None and len(files) > limit:
            raise AssistAIError(
                f"Слишком много изображений: максимум {limit}.",
                error_code="BAD_REQUEST",
            )
        return list(files)
    if os.getenv("ALLOW_COMPOSITE_FALLBACK", "0").strip() == "1":
        return [_composite_edit_canvas(files, labels)]
    raise AssistAIError(
        f"Модель {model_id} не принимает несколько изображений. "
        "Выберите другую в настройках или опишите словами, что взять со второго фото."
    )


def _composite_edit_canvas(
    files: list[tuple[bytes, str, str]],
    labels: list[str] | None,
) -> tuple[bytes, str, str]:
    """Склеивает фото слева направо и подписывает половины номерами."""
    from io import BytesIO

    from PIL import Image, ImageDraw

    opened: list[Image.Image] = []
    try:
        for data, _name, _mime in files:
            opened.append(Image.open(BytesIO(data)).convert("RGB"))
        height = max(image.height for image in opened)
        width = sum(image.width for image in opened)
        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)
        offset = 0
        for index, image in enumerate(opened):
            canvas.paste(image, (offset, 0))
            mark = labels[index] if labels and index < len(labels) else str(index + 1)
            draw.text((offset + 4, 4), mark, fill=(0, 0, 0))
            offset += image.width
        buffer = BytesIO()
        canvas.save(buffer, format="PNG")
        return buffer.getvalue(), "composite.png", "image/png"
    finally:
        for image in opened:
            image.close()


def _is_flux_model(model: str) -> bool:
    """FLUX не принимает пиксельный ``size`` вроде ``1024x1024``."""
    return image_model_family(model) == "flux"


def _snap_aspect_ratio(width: int, height: int) -> str:
    """Ближайшее соотношение сторон из набора, который понимает FLUX."""
    if width <= 0 or height <= 0:
        return "1:1"
    target = width / height
    numer, denom = min(
        _FLUX_ASPECTS,
        key=lambda item: abs((item[0] / item[1]) - target),
    )
    return f"{numer}:{denom}"


def _normalize_aspect_ratio(value: str) -> str | None:
    """``16:9`` / ``16 : 9`` → ``16:9``. Иначе ``None``."""
    match = _ASPECT_RATIO.fullmatch(value.strip())
    if match is None:
        return None
    return f"{int(match.group(1))}:{int(match.group(2))}"


def _flux_size_fields(cleaned: str) -> dict[str, Any]:
    """Ярус FLUX и при необходимости ``aspect_ratio`` из пикселей."""
    known = _FLUX_TIER_NAMES.get(cleaned.upper())
    if known is not None:
        return {"size": known}
    match = _PIXEL_SIZE.fullmatch(cleaned)
    if match is None:
        return {"size": cleaned}
    width = int(match.group(1))
    height = int(match.group(2))
    long_side = max(width, height)
    tier = min(
        _FLUX_SIZE_TIERS,
        key=lambda item: (abs(item[0] - long_side), item[0]),
    )[1]
    fields: dict[str, Any] = {"size": tier}
    ratio = _snap_aspect_ratio(width, height)
    if ratio != "1:1":
        fields["aspect_ratio"] = ratio
    return fields


def _qwen_size_fields(cleaned: str) -> dict[str, Any]:
    """Qwen: ``aspect_ratio``, пиксели дополнительно как ``size``."""
    aspect = _normalize_aspect_ratio(cleaned)
    if aspect is not None:
        return {"aspect_ratio": aspect}
    match = _PIXEL_SIZE.fullmatch(cleaned)
    if match is not None:
        width = int(match.group(1))
        height = int(match.group(2))
        return {
            "aspect_ratio": _snap_aspect_ratio(width, height),
            "size": f"{width}x{height}",
        }
    known = _FLUX_TIER_NAMES.get(cleaned.upper())
    if known is not None:
        return {"aspect_ratio": "1:1"}
    return {"aspect_ratio": cleaned}


def _seedream_size_fields(cleaned: str) -> dict[str, Any]:
    """Seedream: ``size_preset`` или ``width``/``height``, не OpenAI ``size``."""
    known = _FLUX_TIER_NAMES.get(cleaned.upper())
    if known is not None:
        return {"size_preset": known}
    match = _PIXEL_SIZE.fullmatch(cleaned)
    if match is not None:
        return {
            "width": int(match.group(1)),
            "height": int(match.group(2)),
        }
    aspect = _normalize_aspect_ratio(cleaned)
    if aspect is not None:
        return {"size_preset": "1K", "aspect_ratio": aspect}
    return {"size_preset": cleaned}


def _recraft_size_fields(cleaned: str) -> dict[str, Any]:
    """Recraft предпочитает ``size`` как ``w:h``; пиксели сводим к соотношению."""
    aspect = _normalize_aspect_ratio(cleaned)
    if aspect is not None:
        return {"size": aspect}
    match = _PIXEL_SIZE.fullmatch(cleaned)
    if match is not None:
        return {
            "size": _snap_aspect_ratio(int(match.group(1)), int(match.group(2))),
        }
    return {"size": cleaned}


_OPENAI_SIZE_ALIASES = {
    "1024x1792": "1024x1536",
    "1792x1024": "1536x1024",
}
_IMAGE_PARAM_REJECTED = re.compile(
    r"complexity|quality|unknown|unsupported", re.IGNORECASE
)


def _image_params_rejected(exc: AssistAIError, payload: dict[str, Any]) -> bool:
    """True, если 400 похож на отказ от quality/complexity и их ещё можно убрать."""
    if exc.status_code != 400:
        return False
    if "quality" not in payload and "complexity" not in payload:
        return False
    return _IMAGE_PARAM_REJECTED.search(exc.message) is not None


def _image_records(data: dict[str, Any]) -> list[dict[str, str]]:
    """URL или base64 и, если провайдер прислал, ``generation_id``."""
    records: list[dict[str, str]] = []
    raw_items = data.get("data")
    if not isinstance(raw_items, list):
        return records
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        url = item.get("url")
        encoded = item.get("b64_json")
        if isinstance(url, str) and url:
            record = {"type": "url", "value": url}
        elif isinstance(encoded, str) and encoded:
            record = {"type": "base64", "value": encoded}
        else:
            continue
        generation_id = item.get("generation_id")
        if isinstance(generation_id, str) and generation_id:
            record["generation_id"] = generation_id
        records.append(record)
    return records


def _usage_dict(data: dict[str, Any]) -> dict[str, Any] | None:
    usage = data.get("usage")
    return usage if isinstance(usage, dict) else None


def _without_image_params(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"quality", "complexity"}
    }


def _image_generation_body(
    model: str,
    prompt: str,
    size: str,
    *,
    quality: str | None = None,
    complexity: str | None = None,
    send_complexity: bool = True,
) -> dict[str, Any]:
    """Тело ``images/generations`` с учётом семейства модели."""
    fields: dict[str, Any] = {"model": model, "prompt": prompt, "n": 1}
    cleaned = size.strip()
    family = image_model_family(model)
    if family == "openai":
        fields["size"] = _OPENAI_SIZE_ALIASES.get(cleaned, cleaned)
        if quality:
            fields["quality"] = quality
        if complexity and send_complexity:
            fields["complexity"] = complexity
        return fields
    if family == "flux":
        fields.update(_flux_size_fields(cleaned))
        return fields
    if family == "qwen":
        fields.update(_qwen_size_fields(cleaned))
        return fields
    if family == "seedream":
        fields.update(_seedream_size_fields(cleaned))
        return fields
    fields.update(_recraft_size_fields(cleaned))
    return fields


class ImageOpsMixin(ProviderTransportMixin):
    """Миксин generate/edit image для ``AssistAIClient``."""

    def _resolve_edit_size(self, explicit: str | None, first_bytes: bytes) -> str:
        """Размер правки: явный аргумент важнее режима конфига."""
        if isinstance(explicit, str) and explicit.strip():
            return explicit.strip()
        mode = self.config.image_edit_size_mode
        if mode == "fixed":
            return "1024x1024"
        if mode == "auto":
            return "auto"
        from app.media.media_utils import fit_image_size, source_image_size

        try:
            width, height = source_image_size(first_bytes)
        except Exception:
            logger.warning("edit_size_unreadable")
            return "auto"
        return fit_image_size(width, height)

    def _downscale_edit_input(self, data: bytes, mime: str) -> tuple[bytes, str]:
        """Даунскейл байтов правки. Размер запроса считается до этого шага."""
        from app.media.media_utils import downscale_for_edit

        return downscale_for_edit(data, mime, self.config.image_input_max_edge)

    async def _post_image_json(
        self,
        payload: dict[str, Any],
        timeout: httpx.Timeout,
        provider: ProviderConfig | None = None,
    ) -> dict[str, Any]:
        """POST генерации. Один повтор без quality/complexity, если провайдер их отверг."""
        target = provider or self._endpoint("image")
        try:
            return await self._post_json(
                _IMAGES_PATH, payload, timeout=timeout, provider=target
            )
        except AssistAIError as exc:
            if not _image_params_rejected(exc, payload):
                raise
            logger.warning(
                "image_params_rejected",
                model=payload.get("model"),
                status_code=exc.status_code,
            )
            return await self._post_json(
                _IMAGES_PATH,
                _without_image_params(payload),
                timeout=timeout,
                provider=target,
            )

    async def _post_image_edit(
        self,
        form: dict[str, str],
        files: dict[str, tuple[str, bytes, str]]
        | list[tuple[str, tuple[str, bytes, str]]],
        provider: ProviderConfig | None = None,
    ) -> dict[str, Any]:
        """Multipart ``/images/edits`` с одним повтором без quality/complexity."""
        target = provider or self._endpoint("image")
        try:
            return await self._post_edit_once(form, files, target)
        except AssistAIError as exc:
            if not _image_params_rejected(exc, form):
                raise
            logger.warning(
                "image_params_rejected",
                model=form.get("model"),
                status_code=exc.status_code,
            )
            return await self._post_edit_once(
                _without_image_params(form), files, target
            )

    async def _post_edit_once(
        self,
        form: dict[str, str],
        files: dict[str, tuple[str, bytes, str]]
        | list[tuple[str, tuple[str, bytes, str]]],
        provider: ProviderConfig | None = None,
    ) -> dict[str, Any]:
        target = provider or self._endpoint("image")
        label = _provider_label(target)
        url = f"{target.base_url.rstrip('/')}/{_IMAGES_EDIT_PATH}"
        read_timeout = max(self.config.timeout, _IMAGE_READ_TIMEOUT)
        timeout = httpx.Timeout(read_timeout, connect=self.config.timeout)
        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=False,
                verify=_ssl_context(),
                headers={
                    "Authorization": f"Bearer {target.api_key}",
                    "User-Agent": "AssistAIClient/1.0",
                },
            ) as client:
                response = await client.post(url, data=form, files=files)
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = _response_error_text(exc.response)
            raise AssistAIError(
                f"{label} вернул HTTP {exc.response.status_code}: {detail}",
                status_code=exc.response.status_code,
                retry_after=_retry_after_seconds(exc.response),
            ) from exc
        except httpx.TimeoutException as exc:
            raise AssistAIError(
                f"Превышено время ожидания ответа {label} ({_IMAGES_EDIT_PATH}).",
                error_code="TIMEOUT",
            ) from exc
        except httpx.RequestError as exc:
            raise AssistAIError(
                f"Сетевая ошибка при обращении к {label}: {exc}",
                error_code="NETWORK_ERROR",
            ) from exc
        try:
            data = response.json()
        except ValueError as exc:
            raise AssistAIError(f"{label} вернул ответ не в формате JSON.") from exc
        if not isinstance(data, dict):
            raise AssistAIError(f"{label} вернул неожиданный формат ответа.")
        return data

    async def _polza_image(
        self,
        provider: ProviderConfig,
        *,
        model: str,
        prompt: str,
        size: str,
        images: list[bytes] | None = None,
    ) -> dict[str, Any]:
        """Генерация или правка через Polza ``/v1/media`` под breaker провайдера."""
        breaker = get_breaker(provider.id)
        try:
            await breaker.ensure_allows_call()
        except CircuitBreakerError as exc:
            raise AssistAIError(
                "⚠️ Сервис генерации изображений временно недоступен. "
                "Попробуйте через минуту.",
                error_code="CIRCUIT_OPEN",
            ) from exc
        fields = polza_media.polza_size_fields(size)
        read_timeout = max(self.config.timeout, _IMAGE_READ_TIMEOUT)
        timeout = httpx.Timeout(read_timeout, connect=self.config.timeout)
        try:
            result = await polza_media.generate(
                self._client_for(provider),
                provider,
                model=model,
                prompt=prompt,
                aspect_ratio=fields.get("aspect_ratio"),
                size=fields.get("size"),
                quality=self.config.image_quality,
                images=images,
                user=polza_media.hash_user_id(get_billing_user()),
                poll_interval=self.config.polza_poll_interval,
                total_wait=self.config.polza_total_wait,
                timeout=timeout,
            )
        except AssistAIError as exc:
            # Только недоступность считается сбоем breaker-а; 4xx и 429 — нет.
            if counts_toward_circuit(exc):
                await breaker._on_failure(exc)
            raise
        await breaker._on_success()
        return result.as_images_response()

    async def _attempt_image(
        self,
        provider: ProviderConfig,
        attempt: Callable[[ProviderConfig], Awaitable[dict[str, Any]]],
    ) -> dict[str, Any]:
        """Один провайдер: до двух повторов на 429 (``Retry-After`` или 1/2 с)."""
        retries = 0
        while True:
            try:
                return await attempt(provider)
            except AssistAIError as exc:
                if not is_rate_limited(exc) or retries >= _RATE_LIMIT_RETRIES:
                    raise
                delay = (
                    exc.retry_after
                    if exc.retry_after is not None
                    else float(2**retries)
                )
                delay = min(max(delay, 0.0), _MAX_RETRY_AFTER_SECONDS)
                retries += 1
                logger.warning(
                    "provider_rate_limited",
                    provider=provider.id,
                    retry=retries,
                    delay=delay,
                )
                await _sleep(delay)

    async def _image_with_fallback(
        self,
        attempt: Callable[[ProviderConfig], Awaitable[dict[str, Any]]],
    ) -> tuple[dict[str, Any], ProviderConfig]:
        """Картинка у основного провайдера; один fallback только при недоступности."""
        primary, fallback = self._registry.resolve_route("image")
        try:
            return await self._attempt_image(primary, attempt), primary
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
            return await self._attempt_image(fallback, attempt), fallback

    async def _record_image_spend(
        self,
        *,
        count: int,
        model: str,
        size: str,
        data: dict[str, Any],
        provider: ProviderConfig,
    ) -> None:
        estimate = cost_tracker.estimate_image_cost_detailed(
            count,
            model=model,
            quality=self.config.image_quality,
            size=size,
            usage=_usage_dict(data),
            provider=provider,
            rub_per_usd=self.config.rub_per_usd,
        )
        await cost_tracker.record(
            kind="image",
            amount_usd=estimate.amount_usd,
            provider=provider.id,
            model=model,
            source=estimate.source,
            cost_rub=estimate.cost_rub,
        )

    async def generate_image(
        self,
        prompt: str,
        model: str | None = None,
        size: str = "1024x1024",
    ) -> list[str]:
        """Генерирует изображение и возвращает URL или строки base64.

        Запрос уходит основному провайдеру картинок: ProxyAPI
        ``{base_url}/images/generations`` или Polza ``/v1/media``
        (``IMAGE_PROVIDER``). Если элемент ответа
        содержит ``url``, в список попадает он. Иначе, при наличии ``b64_json``,
        возвращается сама строка base64 без префикса data-URL.

        Таймаут чтения для этого запроса не короче 180 секунд: картинка
        готовится дольше текстового ответа. Таймаут соединения остаётся
        ``config.timeout``. Остальные запросы клиента ограничены им целиком.

        Args:
            prompt: Текстовое описание изображения.
            model: Модель генерации. По умолчанию ``default_image_model``.
            size: Размер в формате семейства модели (пиксели, ярус или ``w:h``).
                Преобразование в поля запроса делает ``_image_generation_body``.

        Returns:
            Список URL или base64-строк, по одному элементу на картинку.

        Raises:
            ValueError: Пустой prompt или size.
            AssistAIError: Ошибка HTTP или ответ без изображения.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt не может быть пустым.")
        if not isinstance(size, str) or not size.strip():
            raise ValueError("size не может быть пустым.")
        if not security.validate_prompt_length(prompt):
            raise ValueError("Prompt слишком длинный")
        sanitized_prompt = security.sanitize_prompt(prompt)
        selected_model = model or self.config.default_image_model
        image_cost = cost_tracker.estimate_image_cost(
            1,
            model=selected_model,
            quality=self.config.image_quality,
            size=size,
            provider=self._endpoint("image"),
            rub_per_usd=self.config.rub_per_usd,
        )
        # Бюджет держим здесь (и в edit_images), не в хендлерах: иначе двойная
        # резервация при ARQ/tool-path и ложные отказы у края лимита.
        await cost_tracker.check_budget(estimated=image_cost)
        try:
            payload = _image_generation_body(
                selected_model,
                sanitized_prompt,
                size.strip(),
                quality=self.config.image_quality,
                complexity=self.config.image_complexity_generate,
                send_complexity=self.config.send_complexity,
            )
            read_timeout = max(self.config.timeout, _IMAGE_READ_TIMEOUT)
            timeout = httpx.Timeout(read_timeout, connect=self.config.timeout)
            size_label = (
                payload.get("size")
                or payload.get("size_preset")
                or payload.get("aspect_ratio")
                or size.strip()
            )
            logger.info(
                "generate_image",
                model=selected_model,
                size=size_label,
                prompt_chars=len(sanitized_prompt),
            )
            polza_size = size.strip()
            if image_model_family(selected_model) == "openai":
                polza_size = _OPENAI_SIZE_ALIASES.get(polza_size, polza_size)

            async def attempt(provider: ProviderConfig) -> dict[str, Any]:
                if provider.kind == "polza_media":
                    return await self._polza_image(
                        provider,
                        model=selected_model,
                        prompt=sanitized_prompt,
                        size=polza_size,
                    )
                return await self._post_image_json(
                    payload, timeout=timeout, provider=provider
                )

            data, used = await self._image_with_fallback(attempt)

            records = _image_records(data)
            _set_generation_ids([record.get("generation_id") for record in records])
            images = [record["value"] for record in records]

            if not images:
                logger.error("generate_image: в ответе нет url и b64_json")
                raise AssistAIError(f"{_provider_label(used)} не вернул изображение.")

            logger.info("generate_image: получено изображений: %s", len(images))
            await self._record_image_spend(
                count=len(images),
                model=selected_model,
                size=str(size_label),
                data=data,
                provider=used,
            )
            return images
        finally:
            await cost_tracker.release_reservation()

    async def edit_image(
        self,
        image_bytes: bytes,
        prompt: str,
        *,
        filename: str = "image.png",
        mime_type: str = "image/png",
        model: str | None = None,
    ) -> list[str]:
        """Редактирует изображение через ``/images/edits`` (multipart) или Polza media."""
        if not image_bytes:
            raise ValueError("image_bytes не может быть пустым.")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt не может быть пустым.")
        if not security.validate_prompt_length(prompt):
            raise ValueError("Prompt слишком длинный")
        if self._closed:
            raise AssistAIError("Клиент уже закрыт.")

        selected_model = model or self.config.default_image_edit_model
        edit_size = self._resolve_edit_size(None, image_bytes)
        image_cost = cost_tracker.estimate_image_cost(
            1,
            model=selected_model,
            quality=self.config.image_quality,
            size=edit_size,
            provider=self._endpoint("image"),
            rub_per_usd=self.config.rub_per_usd,
        )
        await cost_tracker.check_budget(estimated=image_cost)
        try:
            image_bytes, mime_type = self._downscale_edit_input(image_bytes, mime_type)
            sanitized = security.sanitize_prompt(prompt.strip())
            body = _image_generation_body(
                selected_model,
                sanitized,
                edit_size,
                quality=self.config.image_quality,
                complexity=self.config.image_complexity_edit,
                send_complexity=self.config.send_complexity,
            )
            form = {key: str(value) for key, value in body.items() if key != "n"}
            files = {
                "image": (
                    filename,
                    image_bytes,
                    mime_type or "application/octet-stream",
                )
            }
            logger.info(
                "edit_image",
                model=selected_model,
                bytes=len(image_bytes),
                prompt_chars=len(sanitized),
            )
            polza_size = _OPENAI_SIZE_ALIASES.get(edit_size, edit_size)

            async def attempt(provider: ProviderConfig) -> dict[str, Any]:
                if provider.kind == "polza_media":
                    return await self._polza_image(
                        provider,
                        model=selected_model,
                        prompt=sanitized,
                        size=polza_size,
                        images=[image_bytes],
                    )
                return await self._post_image_edit(form, files, provider)

            data, used = await self._image_with_fallback(attempt)

            records = _image_records(data)
            _set_generation_ids([record.get("generation_id") for record in records])
            images = [record["value"] for record in records]
            if not images:
                raise AssistAIError(
                    f"{_provider_label(used)} не вернул отредактированное изображение."
                )
            await self._record_image_spend(
                count=len(images),
                model=selected_model,
                size=edit_size,
                data=data,
                provider=used,
            )
            return images
        finally:
            await cost_tracker.release_reservation()

    async def edit_images(
        self,
        files: list[tuple[bytes, str, str]],
        prompt: str,
        *,
        model: str | None = None,
        labels: list[str] | None = None,
        size: str | None = None,
        complexity: str | None = None,
    ) -> list[str]:
        """Правит несколько фото через multipart ``/images/edits`` или Polza media.

        Первый файл — основа. Поле формы — ``image[]``. Размер уходит в полях
        семейства модели, как у ``generate_image``.

        ``complexity`` overrides ``config.image_complexity_edit`` when set (used
        by photo generate-with-refs to keep generate-tier quality).
        """
        if not files:
            raise ValueError("files не может быть пустым.")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt не может быть пустым.")
        if not security.validate_prompt_length(prompt):
            raise ValueError("Prompt слишком длинный")
        if self._closed:
            raise AssistAIError("Клиент уже закрыт.")

        selected_model = model or self.config.default_image_edit_model
        edit_size = self._resolve_edit_size(size, files[0][0])
        edit_complexity = complexity or self.config.image_complexity_edit
        image_cost = cost_tracker.estimate_image_cost(
            1,
            model=selected_model,
            quality=self.config.image_quality,
            size=edit_size,
            provider=self._endpoint("image"),
            rub_per_usd=self.config.rub_per_usd,
        )
        await cost_tracker.check_budget(estimated=image_cost)
        try:
            scaled: list[tuple[bytes, str, str]] = []
            for raw, name, mime in files:
                scaled_data, scaled_mime = self._downscale_edit_input(raw, mime)
                scaled.append((scaled_data, name, scaled_mime))
            polza_size = _OPENAI_SIZE_ALIASES.get(edit_size, edit_size)

            async def attempt(provider: ProviderConfig) -> dict[str, Any]:
                # Лимит и склейка зависят от провайдера, поэтому готовим файлы здесь.
                prepared = prepare_edit_files(
                    scaled, selected_model, labels, provider=provider
                )
                sanitized = security.sanitize_prompt(prompt.strip())
                if labels:
                    numbered = ", ".join(labels)
                    sanitized = f"{sanitized}\nImage order: {numbered}."
                if len(prepared) == 1 and len(files) > 1:
                    sanitized = (
                        f"{sanitized} The canvas is a side-by-side composite. "
                        "The left half is the base image."
                    )
                logger.info(
                    "edit_image",
                    model=selected_model,
                    files=len(prepared),
                    prompt_chars=len(sanitized),
                    provider=provider.id,
                    complexity=edit_complexity,
                )
                if provider.kind == "polza_media":
                    return await self._polza_image(
                        provider,
                        model=selected_model,
                        prompt=sanitized,
                        size=polza_size,
                        images=[data for data, _name, _mime in prepared],
                    )
                body = _image_generation_body(
                    selected_model,
                    sanitized,
                    edit_size,
                    quality=self.config.image_quality,
                    complexity=edit_complexity,
                    send_complexity=self.config.send_complexity,
                )
                form = {
                    key: str(value) for key, value in body.items() if key not in {"n"}
                }
                multipart = [
                    ("image[]", (name, data, mime or "application/octet-stream"))
                    for data, name, mime in prepared
                ]
                return await self._post_image_edit(form, multipart, provider)

            payload, used = await self._image_with_fallback(attempt)
            records = _image_records(payload)
            _set_generation_ids([record.get("generation_id") for record in records])
            images = [record["value"] for record in records]
            if not images:
                raise AssistAIError(
                    f"{_provider_label(used)} не вернул отредактированное изображение."
                )
            await self._record_image_spend(
                count=len(images),
                model=selected_model,
                size=edit_size,
                data=payload,
                provider=used,
            )
            return images
        finally:
            await cost_tracker.release_reservation()
