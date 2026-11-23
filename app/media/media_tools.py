"""Tools персоны для медиа: edit image, video, STT, TTS + исходящие вложения."""

from __future__ import annotations

import json
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any, Literal, Self

import structlog

from app.infra.cost_tracker import BudgetExceededError, cost_tracker, prefer_cost_rub
from app.llm.assist_ai import AssistAIClient
from app.llm.errors import AssistAIError
from app.llm.image_ops import (
    _get_generation_ids,
    current_image_as_file,
    generated_image_records,
    note_generation_id,
)
from app.llm.media_api import (
    VIDEO_MAX_REFERENCE_IMAGES,
    SpeechResult,
    TranscriptionResult,
    VideoReference,
    decode_image_b64,
    safe_image_mime,
)
from app.llm.tooling import BaseTool, _error_json, _exception_text
from app.llm.video_settings import current_video_settings
from app.media.media_utils import MediaProcessingError, prepare_audio_for_stt

logger = structlog.get_logger("media_tools")

OutboundKind = Literal["image", "audio", "video", "document"]
UserMediaKind = Literal["image", "audio", "video"]


@dataclass
class OutboundMedia:
    kind: OutboundKind
    filename: str
    mime_type: str
    data: bytes | None = None
    url: str | None = None
    model: str | None = None


@dataclass
class UserMedia:
    kind: UserMediaKind
    filename: str
    mime_type: str
    data: bytes


@dataclass
class MediaBucket:
    """Собирает исходящие файлы за один ход диалога."""

    items: list[OutboundMedia] = field(default_factory=list)

    def add(self, item: OutboundMedia) -> None:
        self.items.append(item)


_REQUEST_OUTBOUND: ContextVar[MediaBucket | None] = ContextVar(
    "request_outbound_media", default=None
)
_REQUEST_USER_MEDIA: ContextVar[list[UserMedia] | None] = ContextVar(
    "request_user_media", default=None
)


class MediaTurnScope:
    """На время LLM-хода открывает корзину исходящих медиа."""

    def __init__(self) -> None:
        self.bucket = MediaBucket()
        self._token: Token[MediaBucket | None] | None = None

    def __enter__(self) -> Self:
        self._token = _REQUEST_OUTBOUND.set(self.bucket)
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._token is not None:
            _REQUEST_OUTBOUND.reset(self._token)


def set_user_media(items: list[UserMedia]) -> Token[list[UserMedia] | None]:
    return _REQUEST_USER_MEDIA.set(items)


def reset_user_media(token: Token[list[UserMedia] | None]) -> None:
    _REQUEST_USER_MEDIA.reset(token)


def get_user_media() -> list[UserMedia]:
    return list(_REQUEST_USER_MEDIA.get() or [])


def _outbound() -> MediaBucket | None:
    return _REQUEST_OUTBOUND.get()


def _last_user_media(*kinds: UserMediaKind) -> UserMedia | None:
    items = get_user_media()
    for item in reversed(items):
        if item.kind in kinds:
            return item
    return None


class EditImageTool(BaseTool):
    """Редактирование одного или нескольких изображений через ``/images/edits``."""

    def __init__(self, client: AssistAIClient) -> None:
        self._client = client

    @property
    def name(self) -> str:
        return "edit_image"

    @property
    def description(self) -> str:
        return (
            "Изменить присланные фото. Если фото больше одного, индексы обязательны: "
            "не угадывай роли по порядку альбома. "
            "Примеры: «костюм с фото 1 надень на человека с фото 2» — "
            "base_index=1, reference_indexes=[0]; "
            "«лицо с фото 1 замени лицом с фото 2» — base_index=1, reference_indexes=[0]; "
            "«предмет с фото 1 помести на фото 2» — base_index=1, reference_indexes=[0]. "
            "Фото 1 — это индекс 0."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "instruction": {
                    "type": "string",
                    "description": "Что изменить. Только просьба пользователя, без служебного текста.",
                },
                "base_index": {
                    "type": "integer",
                    "description": "Индекс основы с нуля. Обязателен, если фото больше одного.",
                },
                "reference_indexes": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Индексы референсов с нуля. Обязательны, если фото больше одного.",
                },
                "edit_type": {
                    "type": "string",
                    "enum": [
                        "face_swap",
                        "outfit_transfer",
                        "object_insert",
                        "background_replace",
                        "style_transfer",
                        "generic",
                    ],
                    "description": "Тип правки.",
                },
            },
            "required": ["instruction"],
            "additionalProperties": False,
        }

    async def execute(self, **kwargs: Any) -> str:
        instruction = kwargs.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            instruction = kwargs.get("prompt")
        model = kwargs.get("model")
        if not isinstance(instruction, str) or not instruction.strip():
            return _error_json("Параметр instruction обязателен.")
        images = [item for item in get_user_media() if item.kind == "image"]
        if not images:
            return _error_json(
                "Нет изображения во вложении. Попросите пользователя прислать картинку."
            )
        from app.infra.cost_tracker import get_billing_user
        from app.infra.rate_limiter import rate_limiter

        user_id = get_billing_user()
        if user_id is not None and not await rate_limiter.check_image_limit(user_id):
            retry_after = await rate_limiter.get_retry_after(user_id, "image")
            return _error_json(
                f"Слишком много запросов картинок. Подождите {int(retry_after) or 60} секунд."
            )
        base, refs, error = _resolve_edit_indexes(images, kwargs)
        if error is not None:
            return _error_json(error)
        assert base is not None
        ordered = [images[base]]
        labels = [str(base + 1)]
        for index in refs:
            ordered.append(images[index])
            labels.append(str(index + 1))
        files = [(item.data, item.filename, item.mime_type) for item in ordered]
        selected = model.strip() if isinstance(model, str) and model.strip() else None
        edit_type = kwargs.get("edit_type")
        from app.infra.security import security

        prompt = security.sanitize_prompt(instruction.strip())
        if isinstance(edit_type, str) and edit_type.strip():
            prompt = f"{edit_type.strip()}: {prompt}"
        config = getattr(self._client, "config", None)
        fallback = getattr(config, "default_image_edit_model", None)
        used_model = selected or (
            fallback if isinstance(fallback, str) and fallback.strip() else None
        )
        logger.info(
            "edit_image_call",
            count=len(files),
            base_index=base,
            reference_indexes=refs,
            model=selected,
        )
        try:
            import app.infra.job_queue as job_queue
            from app.infra.cost_tracker import BudgetExceededError

            result = await job_queue.edit_images(
                self._client,
                files,
                prompt,
                model=selected,
                labels=labels,
            )
        except BudgetExceededError as exc:
            return _error_json(exc.message)
        except AssistAIError as exc:
            from app.bot.image_retry import image_failure_reply
            from app.infra.cost_tracker import get_billing_user
            from app.storage.upload_sessions import get_upload_request_context

            ctx = get_upload_request_context()
            billed = get_billing_user()
            user_id = ctx.user_id if ctx is not None else billed
            chat_id = ctx.chat_id if ctx is not None else user_id
            if chat_id is not None and user_id is not None:
                return _error_json(
                    await image_failure_reply(
                        chat_id,
                        user_id,
                        exc,
                        as_file=current_image_as_file(used_model),
                    )
                )
            return _error_json(exc.message)
        except (TypeError, ValueError) as exc:
            return _error_json(_exception_text(exc))
        except Exception as exc:
            logger.exception("edit_image_failed")
            return _error_json(_exception_text(exc))

        for generation_id in _get_generation_ids():
            note_generation_id(
                generation_id if isinstance(generation_id, str) else None
            )
        bucket = _outbound()
        for index, image in enumerate(result):
            if image.startswith(("http://", "https://")):
                item = OutboundMedia(
                    kind="image",
                    filename=f"edited_{index + 1}.png",
                    mime_type="image/png",
                    url=image,
                    model=used_model,
                )
            else:
                import base64

                item = OutboundMedia(
                    kind="image",
                    filename=f"edited_{index + 1}.png",
                    mime_type="image/png",
                    data=base64.b64decode(image),
                    model=used_model,
                )
            if bucket is not None:
                bucket.add(item)
        logger.info("edit_image_done", count=len(result), base_index=base)
        return json.dumps(
            {
                "status": "success",
                "image_count": len(result),
                "base_index": base,
                "note": (
                    "Отредактированное изображение уже отправляется пользователю. "
                    "Кратко подтверди результат, без base64 и без URL."
                ),
            },
            ensure_ascii=False,
        )


def _resolve_edit_indexes(
    images: list[UserMedia], kwargs: dict[str, Any]
) -> tuple[int | None, list[int], str | None]:
    """Индексы из аргументов модели. При нескольких фото без индексов — ошибка."""
    raw_base = kwargs.get("base_index")
    raw_refs = kwargs.get("reference_indexes")
    count = len(images)
    if count > 1 and (not isinstance(raw_base, int) or not isinstance(raw_refs, list)):
        return (
            None,
            [],
            "Для нескольких фото укажи base_index и reference_indexes. Не угадывай роли.",
        )
    if count == 1 and not isinstance(raw_base, int):
        return 0, [], None
    if not isinstance(raw_base, int):
        return None, [], "Параметр base_index обязателен."
    if not 0 <= raw_base < count:
        return None, [], "Индекс основы вне диапазона фото."
    refs = _int_list(raw_refs) if raw_refs is not None else []
    if refs is None:
        return None, [], "reference_indexes должен быть списком индексов."
    refs = [index for index in refs if index != raw_base]
    if any(index < 0 or index >= count for index in refs):
        return None, [], "Индекс референса вне диапазона фото."
    return raw_base, refs, None


def _int_list(value: Any) -> list[int] | None:
    if value is None:
        return []
    if not isinstance(value, list):
        return None
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            return None
        result.append(item)
    return result


def video_reference_catalog() -> list[VideoReference]:
    """Фото пользователя, затем кадры generate_image этого хода. Индексы с нуля."""
    items: list[VideoReference] = []
    for media in get_user_media():
        if media.kind == "image" and media.data:
            items.append(
                VideoReference(
                    data=media.data, mime_type=safe_image_mime(media.mime_type)
                )
            )
    for record in generated_image_records():
        if record["type"] == "url":
            items.append(VideoReference(url=record["value"]))
            continue
        data, mime = decode_image_b64(record["value"])
        items.append(VideoReference(data=data, mime_type=mime))
    return items


def resolve_video_references(
    raw: Any,
) -> tuple[list[VideoReference] | None, str | None]:
    """Индексы → референсы. Пустой список оставляет текстовый ролик.

    Ошибка возвращается до вызова провайдера, чтобы не открывать платную задачу.
    """
    if raw is None:
        return None, None
    indexes = _int_list(raw)
    if indexes is None:
        return None, "reference_indexes должен быть списком целых чисел."
    if not indexes:
        return None, None
    if len(indexes) > VIDEO_MAX_REFERENCE_IMAGES:
        return None, (
            f"Можно передать не больше {VIDEO_MAX_REFERENCE_IMAGES} референсов."
        )
    if len(set(indexes)) != len(indexes):
        return None, "reference_indexes не должен повторять один и тот же номер."
    catalog = video_reference_catalog()
    if any(index < 0 or index >= len(catalog) for index in indexes):
        return None, (
            "Номер референса вне диапазона. "
            f"Сейчас доступно картинок: {len(catalog)} (номера с 0)."
        )
    return [catalog[index] for index in indexes], None


class GenerateVideoTool(BaseTool):
    """Генерация видео через ``/v1/videos`` с polling."""

    def __init__(self, client: AssistAIClient) -> None:
        self._client = client

    @property
    def name(self) -> str:
        return "generate_video"

    @property
    def description(self) -> str:
        return (
            "Сгенерировать короткое видео по описанию и картинкам-референсам. "
            "Вызывай, когда пользователь просит ролик, клип или анимацию. "
            "Если внешность, предмет или место уже есть на фото пользователя "
            "или на кадре из generate_image, передай reference_indexes. "
            "Без них сервис видит только текст. "
            "Генерация занимает минуты — предупреди об ожидании только кратко. "
            "Формат, длительность, разрешение и звук уже заданы в настройках персоны."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "Описание сцены и движения. Если переданы референсы, "
                        "называй их «первая картинка», «вторая картинка» "
                        "в порядке reference_indexes."
                    ),
                },
                "reference_indexes": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": (
                        "Номера картинок-референсов. Сначала фото пользователя "
                        "в этом сообщении (0, 1, …), затем кадры generate_image "
                        "этого хода — их номера в поле reference_indexes ответа. "
                        "Порядок массива совпадает с «первая картинка» в prompt. "
                        f"Не больше {VIDEO_MAX_REFERENCE_IMAGES}. "
                        "Без поля уходит только текст."
                    ),
                },
                "model": {
                    "type": "string",
                    "description": "Модель видео. Необязательно.",
                },
            },
            "required": ["prompt"],
            "additionalProperties": False,
        }

    async def execute(self, **kwargs: Any) -> str:
        prompt = kwargs.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            return _error_json("Параметр prompt обязателен.")
        references, ref_error = resolve_video_references(
            kwargs.get("reference_indexes")
        )
        if ref_error:
            return _error_json(ref_error)
        from app.infra.cost_tracker import BudgetExceededError, get_billing_user
        from app.infra.rate_limiter import rate_limiter
        from app.infra.security import security

        user_id = get_billing_user()
        if user_id is not None and not await rate_limiter.check_media_limit(user_id):
            retry_after = await rate_limiter.get_retry_after(user_id, "media")
            return _error_json(
                f"Слишком много медиа-запросов. Подождите {int(retry_after) or 60} секунд."
            )
        prompt = security.sanitize_prompt(prompt.strip())
        model = kwargs.get("model")
        selected = model.strip() if isinstance(model, str) and model.strip() else None
        scoped = current_video_settings()
        duration: int | None
        resolution: str | None
        aspect_ratio: str | None
        generate_audio: bool | None
        if scoped is not None:
            duration = scoped.duration
            resolution = scoped.resolution
            aspect_ratio = scoped.aspect_ratio
            generate_audio = scoped.generate_audio
        else:
            raw_duration = kwargs.get("duration")
            raw_resolution = kwargs.get("resolution")
            duration = raw_duration if isinstance(raw_duration, int) else None
            resolution = raw_resolution if isinstance(raw_resolution, str) else None
            aspect_ratio = None
            generate_audio = None
        call: dict[str, Any] = {
            "model": selected,
            "duration": duration,
            "resolution": resolution,
            "references": references,
        }
        if aspect_ratio is not None:
            call["aspect_ratio"] = aspect_ratio
        if generate_audio is not None:
            call["generate_audio"] = generate_audio
        try:
            # Budget preflight — в AssistAIClient.generate_video.
            video = await self._client.generate_video(prompt, **call)
        except BudgetExceededError as exc:
            return _error_json(exc.message)
        except AssistAIError as exc:
            from app.bot.video_retry import video_failure_reply
            from app.storage.upload_sessions import get_upload_request_context

            ctx = get_upload_request_context()
            user_id = ctx.user_id if ctx is not None else get_billing_user()
            chat_id = ctx.chat_id if ctx is not None else user_id
            if chat_id is not None and user_id is not None:
                return _error_json(await video_failure_reply(chat_id, user_id, exc))
            return _error_json(exc.message)
        except Exception as exc:
            logger.exception("generate_video_failed")
            return _error_json(_exception_text(exc))

        bucket = _outbound()
        if bucket is not None:
            bucket.add(
                OutboundMedia(
                    kind="video",
                    filename=video.filename,
                    mime_type=video.mime_type,
                    data=video.data,
                )
            )
        sent = 0 if references is None else len(references)
        return json.dumps(
            {
                "status": "success",
                "filename": video.filename,
                "bytes": len(video.data),
                "references": sent,
                "note": (
                    "Видео уже отправляется пользователю в Telegram. "
                    "Кратко подтверди готовность."
                ),
            },
            ensure_ascii=False,
        )


class TranscribeMediaTool(BaseTool):
    """Распознавание речи из аудио/видео; длинный текст уходит файлом."""

    def __init__(self, client: AssistAIClient) -> None:
        self._client = client

    @property
    def name(self) -> str:
        return "transcribe_media"

    @property
    def description(self) -> str:
        return (
            "Извлечь текст из присланного аудио или видео. "
            "Для видео сначала извлекается звуковая дорожка. "
            "Вызывай, когда пользователь просит расшифровать, распознать речь, "
            "сделать транскрипт или субтитры. "
            "Готовый текст пользователь получает файлом, не вставляй его целиком в ответ."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "language": {
                    "type": "string",
                    "description": "Код языка, например ru. Необязательно.",
                },
            },
            "required": [],
            "additionalProperties": False,
        }

    async def execute(self, **kwargs: Any) -> str:
        language = kwargs.get("language")
        lang = (
            language.strip() if isinstance(language, str) and language.strip() else None
        )
        source = _last_user_media("audio", "video")
        if source is None:
            return _error_json(
                "Нет аудио или видео во вложении. Попросите пользователя прислать файл."
            )
        from app.infra.cost_tracker import get_billing_user
        from app.infra.rate_limiter import rate_limiter

        user_id = get_billing_user()
        if user_id is not None and not await rate_limiter.check_media_limit(user_id):
            retry_after = await rate_limiter.get_retry_after(user_id, "media")
            return _error_json(
                f"Слишком много медиа-запросов. Подождите {int(retry_after) or 60} секунд."
            )
        # Rough duration from payload size (~4 KB/s compressed audio).
        estimated = cost_tracker.estimate_stt_cost(max(15.0, len(source.data) / 4000.0))
        try:
            await cost_tracker.check_budget(estimated=estimated)
        except BudgetExceededError as exc:
            return _error_json(exc.message)
        try:
            chunks = await prepare_audio_for_stt(
                source.data,
                filename=source.filename,
                is_video=source.kind == "video",
            )
            parts: list[str] = []
            facts: list[tuple[float | None, str | None, float | None]] = []
            providers: list[str | None] = []
            models: list[str | None] = []
            saw_unpriced = False
            for audio_bytes, name in chunks:
                raw = await self._client.transcribe_audio(
                    audio_bytes,
                    filename=name,
                    language=lang,
                )
                if isinstance(raw, TranscriptionResult):
                    text = raw.text
                    facts.append((raw.cost_usd, raw.cost_source, raw.cost_rub))
                    providers.append(raw.provider_id)
                    models.append(raw.model)
                else:
                    text = raw if isinstance(raw, str) else str(raw)
                    saw_unpriced = True
                if text:
                    parts.append(text)
            transcript = "\n\n".join(parts).strip()
            if not transcript:
                return _error_json("Речь не распознана.")

            if saw_unpriced:
                amount, cost_source, rub = estimated, None, None
                provider = None
                billed_model = None
            else:
                amount, cost_source, rub = prefer_cost_rub(facts, estimated)
                provider = (
                    providers[0] if providers and len(set(providers)) == 1 else None
                )
                billed_model = models[0] if models and len(set(models)) == 1 else None
            await cost_tracker.record(
                kind="stt",
                amount_usd=amount,
                provider=provider,
                model=billed_model,
                source=cost_source,
                cost_rub=rub,
            )
            bucket = _outbound()
            if bucket is not None:
                bucket.add(
                    OutboundMedia(
                        kind="document",
                        filename="transcript.txt",
                        mime_type="text/plain",
                        data=transcript.encode("utf-8"),
                    )
                )
            preview = transcript[:240] + ("…" if len(transcript) > 240 else "")
            return json.dumps(
                {
                    "status": "success",
                    "chars": len(transcript),
                    "chunks": len(chunks),
                    "preview": preview,
                    "note": (
                        "Полный текст отправлен пользователю файлом transcript.txt. "
                        "Не копируй расшифровку в ответ целиком — кратко подтверди."
                    ),
                },
                ensure_ascii=False,
            )
        except MediaProcessingError as exc:
            return _error_json(exc.message)
        except AssistAIError as exc:
            return _error_json(exc.message)
        except Exception as exc:
            logger.exception("transcribe_media_failed")
            return _error_json(_exception_text(exc))
        finally:
            await cost_tracker.release_reservation()


class RequestLargeMediaUploadTool(BaseTool):
    """Выдаёт одноразовую ссылку на загрузку большого аудио/видео в S3."""

    def __init__(self, client: AssistAIClient) -> None:
        self._client = client

    @property
    def name(self) -> str:
        return "request_large_media_upload"

    @property
    def description(self) -> str:
        return (
            "Создать одноразовую ссылку для загрузки большого аудио/видео в облако "
            "(когда файл не влезает в лимит Telegram ~20 МБ или пользователь просит "
            "ссылку на загрузку). После загрузки бот сам скачает файл и сделает "
            "расшифровку. Обязательно отдай пользователю поле upload_url целиком."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "media_kind": {
                    "type": "string",
                    "enum": ["audio", "video", "auto"],
                    "description": "Ожидаемый тип файла. По умолчанию auto.",
                },
                "note": {
                    "type": "string",
                    "description": "Краткий контекст запроса (необязательно).",
                },
            },
            "required": [],
            "additionalProperties": False,
        }

    async def execute(self, **kwargs: Any) -> str:
        from app.media.upload_server import is_upload_feature_ready, public_upload_url
        from app.storage.upload_sessions import get_upload_request_context

        if not is_upload_feature_ready():
            return _error_json(
                "Загрузка больших файлов не настроена "
                "(нужны YC_* ключи бакета и UPLOAD_PUBLIC_BASE_URL)."
            )
        ctx = get_upload_request_context()
        if ctx is None:
            return _error_json(
                "Нет контекста чата для выдачи ссылки. Попросите пользователя "
                "написать сообщение ещё раз."
            )
        media_kind = kwargs.get("media_kind") or "auto"
        if media_kind not in {"audio", "video", "auto"}:
            media_kind = "auto"

        store = getattr(self._client, "upload_session_store", None)
        if store is None:
            return _error_json("Сервис загрузки не инициализирован.")

        session = await store.create(
            chat_id=ctx.chat_id,
            user_id=ctx.user_id,
            purpose="transcribe",
            media_kind=media_kind,
        )
        url = public_upload_url(session.upload_id, session.token)
        if not url:
            return _error_json("UPLOAD_PUBLIC_BASE_URL не задан.")

        return json.dumps(
            {
                "status": "success",
                "upload_url": url,
                "expires_minutes": 60,
                "media_kind": media_kind,
                "note": (
                    "Обязательно отправь пользователю upload_url одной строкой. "
                    "Попроси открыть ссылку, выбрать файл и дождаться «Готово». "
                    "После загрузки бот сам начнёт расшифровку — не вызывай "
                    "transcribe_media, пока файла ещё нет."
                ),
            },
            ensure_ascii=False,
        )


class SynthesizeSpeechTool(BaseTool):
    """Озвучка текста через TTS."""

    def __init__(self, client: AssistAIClient) -> None:
        self._client = client

    @property
    def name(self) -> str:
        return "synthesize_speech"

    @property
    def description(self) -> str:
        return (
            "Озвучить текст голосом (text-to-speech). "
            "Вызывай, когда пользователь просит озвучить, прочитать вслух "
            "или сделать голосовую версию текста."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Текст для озвучки.",
                },
                "voice": {
                    "type": "string",
                    "description": "Голос TTS, например alloy.",
                },
            },
            "required": ["text"],
            "additionalProperties": False,
        }

    async def execute(self, **kwargs: Any) -> str:
        text = kwargs.get("text")
        if not isinstance(text, str) or not text.strip():
            return _error_json("Параметр text обязателен.")
        voice = kwargs.get("voice")
        selected_voice = (
            voice.strip() if isinstance(voice, str) and voice.strip() else None
        )
        cleaned = text.strip()
        from app.infra.cost_tracker import get_billing_user
        from app.infra.rate_limiter import rate_limiter

        user_id = get_billing_user()
        if user_id is not None and not await rate_limiter.check_media_limit(user_id):
            retry_after = await rate_limiter.get_retry_after(user_id, "media")
            return _error_json(
                f"Слишком много медиа-запросов. Подождите {int(retry_after) or 60} секунд."
            )
        estimated = cost_tracker.estimate_tts_cost(cleaned)
        try:
            await cost_tracker.check_budget(estimated=estimated)
        except BudgetExceededError as exc:
            return _error_json(exc.message)
        try:
            speech: SpeechResult = await self._client.synthesize_speech(
                text=cleaned,
                voice=selected_voice,
            )
            amount, source, rub = prefer_cost_rub(
                [(speech.cost_usd, speech.cost_source, speech.cost_rub)],
                estimated,
            )
            await cost_tracker.record(
                kind="tts",
                amount_usd=amount,
                provider=speech.provider_id,
                model=speech.model,
                source=source,
                cost_rub=rub,
            )
            bucket = _outbound()
            if bucket is not None:
                bucket.add(
                    OutboundMedia(
                        kind="audio",
                        filename=speech.filename,
                        mime_type=speech.mime_type,
                        data=speech.audio,
                    )
                )
            return json.dumps(
                {
                    "status": "success",
                    "chars": len(cleaned),
                    "filename": speech.filename,
                    "note": (
                        "Аудио уже отправляется пользователю. "
                        "Кратко подтверди, что озвучка готова."
                    ),
                },
                ensure_ascii=False,
            )
        except AssistAIError as exc:
            return _error_json(exc.message)
        except Exception as exc:
            logger.exception("synthesize_speech_tool_failed")
            return _error_json(_exception_text(exc))
        finally:
            await cost_tracker.release_reservation()


def register_media_tools(client: AssistAIClient) -> None:
    """Регистрирует медиа-инструменты на клиенте."""
    client.register_tool(EditImageTool(client))
    client.register_tool(GenerateVideoTool(client))
    client.register_tool(TranscribeMediaTool(client))
    client.register_tool(SynthesizeSpeechTool(client))
    client.register_tool(RequestLargeMediaUploadTool(client))
