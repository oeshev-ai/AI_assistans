"""Tools персоны для медиа: edit image, video, STT, TTS + исходящие вложения."""

from __future__ import annotations

import json
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any, Literal, Self

import structlog

from app.llm.assist_ai import (
    AssistAIClient,
    AssistAIError,
    BaseTool,
    SpeechResult,
    _error_json,
    _exception_text,
)
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
    """Редактирование изображения через ``/v1/images/edits``."""

    def __init__(self, client: AssistAIClient) -> None:
        self._client = client

    @property
    def name(self) -> str:
        return "edit_image"

    @property
    def description(self) -> str:
        return (
            "Изменить уже существующее изображение по текстовому описанию. "
            "Вызывай, когда пользователь просит отредактировать, перерисовать, "
            "изменить стиль или детали картинки, которую он прислал. "
            "Нужно вложение-изображение в текущем сообщении."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Что изменить на изображении.",
                },
                "model": {
                    "type": "string",
                    "description": "Модель редактирования. Необязательно.",
                },
            },
            "required": ["prompt"],
            "additionalProperties": False,
        }

    async def execute(self, **kwargs: Any) -> str:
        prompt = kwargs.get("prompt")
        model = kwargs.get("model")
        if not isinstance(prompt, str) or not prompt.strip():
            return _error_json("Параметр prompt обязателен.")
        source = _last_user_media("image")
        if source is None:
            return _error_json(
                "Нет изображения во вложении. Попросите пользователя прислать картинку."
            )
        selected = model.strip() if isinstance(model, str) and model.strip() else None
        try:
            images = await self._client.edit_image(
                image_bytes=source.data,
                filename=source.filename,
                prompt=prompt.strip(),
                model=selected,
                mime_type=source.mime_type,
            )
        except AssistAIError as exc:
            return _error_json(exc.message)
        except (TypeError, ValueError) as exc:
            return _error_json(_exception_text(exc))
        except Exception as exc:
            logger.exception("edit_image_failed")
            return _error_json(_exception_text(exc))

        bucket = _outbound()
        for index, image in enumerate(images):
            if image.startswith(("http://", "https://")):
                item = OutboundMedia(
                    kind="image",
                    filename=f"edited_{index + 1}.png",
                    mime_type="image/png",
                    url=image,
                )
            else:
                import base64

                item = OutboundMedia(
                    kind="image",
                    filename=f"edited_{index + 1}.png",
                    mime_type="image/png",
                    data=base64.b64decode(image),
                )
            if bucket is not None:
                bucket.add(item)
        return json.dumps(
            {
                "status": "success",
                "image_count": len(images),
                "note": (
                    "Отредактированное изображение уже отправляется пользователю. "
                    "Кратко подтверди результат, без base64."
                ),
            },
            ensure_ascii=False,
        )


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
            "Сгенерировать короткое видео по текстовому описанию. "
            "Вызывай, когда пользователь просит ролик, клип или анимацию. "
            "Генерация занимает минуты — предупреди об ожидании только кратко."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Описание сцены и движения камеры.",
                },
                "model": {
                    "type": "string",
                    "description": "Модель видео. Необязательно.",
                },
                "duration": {
                    "type": "integer",
                    "description": "Длительность в секундах (если модель поддерживает).",
                },
                "resolution": {
                    "type": "string",
                    "description": "Например 720p.",
                },
            },
            "required": ["prompt"],
            "additionalProperties": False,
        }

    async def execute(self, **kwargs: Any) -> str:
        prompt = kwargs.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            return _error_json("Параметр prompt обязателен.")
        model = kwargs.get("model")
        selected = model.strip() if isinstance(model, str) and model.strip() else None
        duration = kwargs.get("duration")
        resolution = kwargs.get("resolution")
        try:
            video = await self._client.generate_video(
                prompt=prompt.strip(),
                model=selected,
                duration=duration if isinstance(duration, int) else None,
                resolution=resolution if isinstance(resolution, str) else None,
            )
        except AssistAIError as exc:
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
        return json.dumps(
            {
                "status": "success",
                "filename": video.filename,
                "bytes": len(video.data),
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
        try:
            chunks = await prepare_audio_for_stt(
                source.data,
                filename=source.filename,
                is_video=source.kind == "video",
            )
            parts: list[str] = []
            for audio_bytes, name in chunks:
                text = await self._client.transcribe_audio(
                    audio_bytes,
                    filename=name,
                    language=lang,
                )
                if text:
                    parts.append(text)
            transcript = "\n\n".join(parts).strip()
        except MediaProcessingError as exc:
            return _error_json(exc.message)
        except AssistAIError as exc:
            return _error_json(exc.message)
        except Exception as exc:
            logger.exception("transcribe_media_failed")
            return _error_json(_exception_text(exc))

        if not transcript:
            return _error_json("Речь не распознана.")

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
        try:
            speech: SpeechResult = await self._client.synthesize_speech(
                text=text.strip(),
                voice=selected_voice,
            )
        except AssistAIError as exc:
            return _error_json(exc.message)
        except Exception as exc:
            logger.exception("synthesize_speech_tool_failed")
            return _error_json(_exception_text(exc))

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
                "chars": len(text.strip()),
                "filename": speech.filename,
                "note": (
                    "Аудио уже отправляется пользователю. "
                    "Кратко подтверди, что озвучка готова."
                ),
            },
            ensure_ascii=False,
        )


def register_media_tools(client: AssistAIClient) -> None:
    """Регистрирует медиа-инструменты на клиенте."""
    client.register_tool(EditImageTool(client))
    client.register_tool(GenerateVideoTool(client))
    client.register_tool(TranscribeMediaTool(client))
    client.register_tool(SynthesizeSpeechTool(client))
    client.register_tool(RequestLargeMediaUploadTool(client))
