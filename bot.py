"""
Production-ready Telegram-бот с интеграцией AssistAIClient.

Клиент LLM создаётся один раз в startup и живёт в ``workflow_data``.
Цикл Function Calling уже внутри ``stream_chat`` и здесь не повторяется.
Ответ языковой модели приходит потоком и сразу правится в сообщении Telegram.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import json
import logging
import os
import sys
import time
from collections.abc import Awaitable, Callable
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any, Literal

import httpx
import sentry_sdk
import structlog
from aiogram import Bot, Dispatcher, F
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.dispatcher.middlewares.base import BaseMiddleware
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.base import BaseStorage
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.methods import SendMessage
from aiogram.methods.base import TelegramMethod
from aiogram.types import (
    BotCommand,
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    MenuButtonCommands,
    Message,
    ReplyKeyboardMarkup,
    TelegramObject,
)
from dotenv import load_dotenv
from pydantic import ValidationError
from redis.asyncio import Redis
from sentry_sdk.integrations.aiohttp import AioHttpIntegration
from sentry_sdk.integrations.logging import LoggingIntegration

import job_queue
from Assist_ai import (
    AssistAIClient,
    AssistAIConfig,
    AssistAIError,
    ImageModelScope,
    image_model_family,
)
from assistant_storage import (
    DEFAULT_ASSISTANT_ID,
    DEFAULT_ASSISTANT_NAME,
    AssistantProfile,
    AssistantStorage,
    AssistantStorageError,
    compose_system_prompt,
    create_assistant_storage,
)
from cost_tracker import BudgetExceededError, cost_tracker, set_billing_user
from health_server import start_health_server
from history_storage import RedisConfig
from logging_setup import (
    bind_request_context,
    clear_request_context,
    configure_logging,
)
from media_tools import (
    MediaTurnScope,
    OutboundMedia,
    UserMedia,
    reset_user_media,
    set_user_media,
)
from media_utils import MediaProcessingError, prepare_audio_for_stt
from metrics import note_active_user, observe_message, start_metrics_server
from models_config import AVAILABLE_IMAGE_MODELS, AVAILABLE_TEXT_MODELS
from rate_limiter import rate_limiter
from s3_storage import S3Storage, S3StorageError, get_storage
from security import security
from topic_history import (
    TopicConfig,
    TopicHistoryStorage,
    TopicStorageError,
    create_topic_storage,
)
from upload_server import is_upload_feature_ready, setup_upload_routes
from upload_sessions import (
    UploadRequestContext,
    UploadSession,
    UploadSessionStore,
    reset_upload_request_context,
    set_upload_request_context,
)
from vision import (
    MAX_IMAGE_SIZE_MB,
    VisionError,
    build_image_content,
    build_multimodal_content,
    download_telegram_file,
    ensure_within_size_limit,
    extract_image_from_message,
    process_pdf,
)

configure_logging()
logger = structlog.get_logger("TelegramBot")

TELEGRAM_TEXT_LIMIT = 4096
STREAM_EDIT_INTERVAL = 0.5
PAGE_SIZE = 10
WELCOME = (
    "Привет. Я ассистент: отвечаю на обычные вопросы и умею получать актуальные данные.\n\n"
    "Погода: текущая, прогноз, качество воздуха, сравнение городов.\n"
    "Валюты: конвертация суммы из одной валюты в другую.\n"
    "Просто пишите или присылайте файлы — персона сама вызовет нужные действия:\n"
    "картинки (генерация/правка), видео, расшифровка аудио/видео, озвучка текста,\n"
    "погода и валюты.\n\n"
    "Голосовое сообщение распознаётся как ваш текст и уходит в диалог.\n"
    "Длинное аудио/видео для расшифровки пришлите файлом — текст вернётся .txt.\n"
    "Файл больше ~20 МБ: напишите персоне «нужна ссылка на загрузку» — она выдаст "
    "одноразовую ссылку в облако, после загрузки расшифровка стартует сама.\n\n"
    "Внизу кнопки: ассистенты, темы, картинка, настройки.\n"
    "/help — справка, /cancel — отмена мастера"
)
InputKind = Literal["text", "voice"]
MAX_VOICE_DURATION_SECONDS = 600
MAX_VOICE_SIZE_MB = 20
MAX_MEDIA_SIZE_MB = 20  # лимит Telegram Bot API getFile
SHOW_RECOGNIZED_TEXT = True
VOICE_RECOGNIZED_PREVIEW_LIMIT = 500
TTS_MAX_CHARS = 3500
TTS_VOICES: tuple[str, ...] = ("alloy", "verse", "nova", "shimmer")
STT_LANGUAGES: tuple[tuple[str, str | None], ...] = (
    ("авто", None),
    ("ru", "ru"),
    ("en", "en"),
)

_MENU_ASSISTANTS = "Ассистенты"
_MENU_TOPICS = "Темы"
_MENU_SETTINGS = "Настройки"
MAIN_MENU_TEXT = {
    "assistants": "🤖 Ассистенты",
    "topics": "💬 Темы",
    "image": "🖼 Картинка",
    "settings": "⚙️ Настройки",
}
_OPENAI_IMAGE_SIZES: tuple[str, ...] = ("1024x1024", "1024x1792", "1792x1024")
_TIER_IMAGE_SIZES: tuple[str, ...] = ("1K", "2K")
_ASPECT_IMAGE_SIZES: tuple[str, ...] = ("1:1", "16:9", "9:16")
IMAGE_SIZES: tuple[str, ...] = (
    *_OPENAI_IMAGE_SIZES,
    *_TIER_IMAGE_SIZES,
    *_ASPECT_IMAGE_SIZES,
)
_FLOW_KEYS = (
    "flow_name",
    "flow_prompt",
    "flow_edit_id",
    "flow_text_model",
    "flow_image_model",
    "flow_topic_id",
    "flow_topic_name",
    "image_model",
    "image_size",
)
_SCREEN_ASSISTANTS = "screen_assistants"
_SCREEN_TOPICS = "screen_topics"
_SCREEN_SETTINGS = "screen_settings"
_SCREEN_PICKER = "screen_picker"
_SCREEN_IMAGE = "screen_image"
_MENU_ANCHOR = "menu_anchor"
_CHROME = "chrome_ids"
_CHROME_BUCKET: ContextVar[list[int] | None] = ContextVar("chrome_bucket", default=None)
_KEEP_OUTGOING: ContextVar[bool] = ContextVar("keep_outgoing", default=False)
_SCREEN_KEYS = (
    _SCREEN_ASSISTANTS,
    _SCREEN_TOPICS,
    _SCREEN_SETTINGS,
    _SCREEN_PICKER,
    _SCREEN_IMAGE,
)
_VOICE_MODE_LABELS: dict[str, str] = {
    "off": "выкл",
    "auto": "авто",
    "always": "всегда",
}


class NewAssistantSG(StatesGroup):
    """Шаги создания новой персоны."""

    name = State()
    prompt = State()
    models = State()


class EditAssistantSG(StatesGroup):
    """Правка имени или системного промпта существующей персоны."""

    name = State()
    prompt = State()


class NewTopicSG(StatesGroup):
    """Шаг ввода названия новой темы."""

    name = State()


class RenameTopicSG(StatesGroup):
    """Шаг ввода нового названия темы."""

    name = State()


class ImageSG(StatesGroup):
    """Экран выбора модели и размера, затем описание картинки."""

    config = State()
    prompt = State()


MSG_UNAVAILABLE = "⚠️ Сервис временно недоступен. Попробуйте позже."
MSG_TIMEOUT = "⚠️ Превышено время ожидания."
MSG_INTERNAL = "❌ Произошла внутренняя ошибка."
MSG_IMAGE_FAILED = "Не удалось сгенерировать изображение."
MSG_HISTORY = "⚠️ Временные проблемы с историей диалога. Попробуйте позже."
MSG_PROFILE = "⚠️ Не удалось прочитать профиль ассистента. Попробуйте позже."


async def _active_profile(chat_id: int, storage: AssistantStorage) -> AssistantProfile:
    """Возвращает активную персону. Пустой чат получает «Стандартный»."""
    profile = await storage.get_active_assistant(chat_id)
    if profile is None:
        raise AssistantStorageError("Активный ассистент не найден.")
    return profile


def _text_model_of(profile: AssistantProfile, assistant: AssistAIClient) -> str:
    """Модель диалога: выбор профиля или дефолт клиента."""
    return profile.text_model or assistant.config.default_text_model


def _image_model_of(profile: AssistantProfile, assistant: AssistAIClient) -> str:
    """Модель картинки: выбор профиля или дефолт клиента."""
    return profile.image_model or assistant.config.default_image_model


def _menu_button(title: str, name: str) -> str:
    """Подпись нижней кнопки с именем. Telegram режет текст длиннее 64 символов."""
    prefix = f"{title} ("
    suffix = ")"
    room = 64 - len(prefix) - len(suffix)
    if room < 1:
        return title[:64]
    shown = name if len(name) <= room else f"{name[: room - 3]}..."
    return f"{prefix}{shown}{suffix}"


def _reply_menu(assistant_name: str, topic_name: str) -> ReplyKeyboardMarkup:
    """Три кнопки: персоны, темы активной персоны и её настройки."""
    return ReplyKeyboardMarkup(
        keyboard=[
            [
                KeyboardButton(text=_menu_button(_MENU_ASSISTANTS, assistant_name)),
                KeyboardButton(text=_menu_button(_MENU_TOPICS, topic_name)),
            ],
            [KeyboardButton(text=_MENU_SETTINGS)],
        ],
        resize_keyboard=True,
        is_persistent=True,
    )


def _size_choices(model_id: str) -> tuple[str, ...]:
    """Кнопки размера зависят от семейства image-модели."""
    family = image_model_family(model_id)
    if family in {"flux", "seedream"}:
        return _TIER_IMAGE_SIZES
    if family in {"qwen", "recraft"}:
        return _ASPECT_IMAGE_SIZES
    return _OPENAI_IMAGE_SIZES


def _default_image_size(model_id: str) -> str:
    family = image_model_family(model_id)
    if family in {"flux", "seedream"}:
        return "1K"
    if family in {"qwen", "recraft"}:
        return "1:1"
    return "1024x1024"


def _display_image_size(profile: AssistantProfile, assistant: AssistAIClient) -> str:
    """Размер для экрана настроек. Пустой или чужой семье размер показывает дефолт."""
    model_id = _image_model_of(profile, assistant)
    if profile.image_size in _size_choices(model_id):
        return profile.image_size or _default_image_size(model_id)
    return _default_image_size(model_id)


def _stored_image_size(profile: AssistantProfile) -> str:
    """Размер для генерации. Пустое поле персоны остаётся 1024x1024."""
    if isinstance(profile.image_size, str) and profile.image_size.strip():
        return profile.image_size.strip()
    return "1024x1024"


async def _menu_for(
    chat_id: int,
    assistant_storage: AssistantStorage,
    topic_storage: TopicHistoryStorage,
    *,
    ensure_topic: bool = False,
) -> ReplyKeyboardMarkup:
    """Собирает нижнее меню по активной персоне и её теме."""
    assistant_name = DEFAULT_ASSISTANT_NAME
    assistant_id = DEFAULT_ASSISTANT_ID
    try:
        profile = await _active_profile(chat_id, assistant_storage)
        assistant_name = profile.name
        assistant_id = profile.id
    except AssistantStorageError:
        logger.exception("Меню без профиля, chat_id=%s", chat_id)
    topic_name = "General"
    try:
        if ensure_topic:
            topic_id = await topic_storage.ensure_active(
                chat_id, assistant_id, "General"
            )
        else:
            topic_id = await topic_storage.get_active_topic(chat_id, assistant_id)
        topics = await topic_storage.list_topics(chat_id, assistant_id)
        found = next(
            (item["name"] for item in topics if item["id"] == topic_id),
            "",
        )
        if found:
            topic_name = found
    except TopicStorageError:
        logger.exception("Меню без темы, chat_id=%s", chat_id)
    return _reply_menu(assistant_name, topic_name)


async def _prepare_messages(
    chat_id: int,
    assistant_id: str,
    user_text: str,
    topic_id: str,
    storage: TopicHistoryStorage,
    profile: AssistantProfile,
) -> list[dict[str, Any]]:
    """Добавляет реплику пользователя и собирает запрос для активной персоны.

    В модель уходит системный промпт профиля и история темы без старых
    ``system``. ``stream_chat`` дописывает tool-сообщения в переданный список,
    поэтому это копия: черновики не попадают в Redis.
    """
    limit = storage.max_length
    await storage.append_message(
        chat_id,
        assistant_id,
        topic_id,
        {"role": "user", "content": user_text},
        limit,
    )
    stored = await storage.get_history(chat_id, assistant_id, topic_id)
    dialog = [dict(item) for item in stored if item.get("role") != "system"]
    if len(dialog) != len(stored):
        # Промпт персоны подставляется только в запрос. Старый system в теме
        # занимает слот trim и расходится с активным ассистентом — убираем.
        await storage.clear_topic(chat_id, assistant_id, topic_id)
        for item in dialog:
            await storage.append_message(chat_id, assistant_id, topic_id, item, limit)
    return [
        {"role": "system", "content": compose_system_prompt(profile.system_prompt)},
        *dialog,
    ]


async def _prepare_messages_with_image(
    chat_id: int,
    assistant_id: str,
    multimodal_content: list[dict[str, Any]],
    history_placeholder: str,
    topic_id: str,
    storage: TopicHistoryStorage,
    profile: AssistantProfile,
) -> list[dict[str, Any]]:
    """Пишет текстовый заменитель в тему, а в модель отдаёт multimodal user."""
    limit = storage.max_length
    await storage.append_message(
        chat_id,
        assistant_id,
        topic_id,
        {"role": "user", "content": history_placeholder},
        limit,
    )
    stored = await storage.get_history(chat_id, assistant_id, topic_id)
    dialog = [dict(item) for item in stored if item.get("role") != "system"]
    if len(dialog) != len(stored):
        await storage.clear_topic(chat_id, assistant_id, topic_id)
        for item in dialog:
            await storage.append_message(chat_id, assistant_id, topic_id, item, limit)
    prior = dialog[:-1] if dialog else []
    return [
        {
            "role": "system",
            "content": compose_system_prompt(profile.system_prompt),
        },
        *prior,
        {"role": "user", "content": multimodal_content},
    ]


def _vision_model_of(profile: AssistantProfile, assistant: AssistAIClient) -> str:
    """Модель анализа картинок: vision → text → дефолт клиента."""
    return (
        profile.vision_model
        or profile.text_model
        or assistant.config.default_text_model
    )


def _compact_tool_content(content: str) -> str:
    """Убирает URL и base64 из результата generate_image, оставляя факт успеха."""
    try:
        payload = json.loads(content)
    except ValueError:
        return content
    if not isinstance(payload, dict):
        return content
    raw_images = payload.get("images")
    if not isinstance(raw_images, list):
        return content
    compacted = dict(payload)
    compacted["images"] = [
        {"type": item.get("type")}
        if isinstance(item, dict) and "type" in item
        else {"type": "unknown"}
        for item in raw_images
    ]
    if "status" not in compacted:
        compacted["status"] = "success"
    if "note" not in compacted:
        compacted["note"] = (
            "Изображение уже передано пользователю в Telegram. "
            "Не вызывай generate_image повторно для этого запроса."
        )
    return json.dumps(compacted, ensure_ascii=False)


def _sanitize_history_message(message: dict[str, Any]) -> dict[str, Any] | None:
    """Готовит сообщение раунда модели к записи в тему.

    System и user не сохраняем здесь: user уже записан в ``_prepare_messages``.
    Reasoning убираем, чтобы не тащить его в следующие запросы других моделей.
    """
    role = message.get("role")
    if role not in {"assistant", "tool"}:
        return None
    stored = {
        key: value
        for key, value in message.items()
        if key not in {"reasoning", "reasoning_content"}
    }
    if role == "tool" and isinstance(stored.get("content"), str):
        stored["content"] = _compact_tool_content(stored["content"])
    return stored


def _image_prompts_from_messages(messages: list[dict[str, Any]]) -> list[str]:
    """Достаёт prompt из коротких JSON-ответов generate_image."""
    prompts: list[str] = []
    for item in messages:
        if item.get("role") != "tool":
            continue
        content = item.get("content")
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except ValueError:
            continue
        if not isinstance(payload, dict):
            continue
        prompt = payload.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            prompts.append(prompt.strip())
    return prompts


def _image_completion_text(messages: list[dict[str, Any]], image_count: int) -> str:
    """Текст-заглушка, если модель сгенерировала картинку без финальной реплики."""
    prompts = _image_prompts_from_messages(messages)
    if len(prompts) == 1:
        subject = f"по запросу: «{prompts[0]}»"
    elif prompts:
        joined = "; ".join(prompts[:3])
        subject = f"по запросам: «{joined}»"
    else:
        subject = f"({image_count} шт.)"
    return (
        f"Сгенерировано изображение {subject}. "
        "Картинка уже отправлена пользователю в Telegram. "
        "Повторно вызывать generate_image для этого запроса не нужно."
    )


async def _remember_model_turn(
    chat_id: int,
    assistant_id: str,
    topic_id: str,
    storage: TopicHistoryStorage,
    new_messages: list[dict[str, Any]],
    *,
    answer: str,
    images: list[dict[str, Any]],
) -> None:
    """Пишет в тему tool-раунд и финальный ответ без сырых картинок.

    Без этого при возврате к теме модель снова вызывает generate_image:
    в истории был только user, а успех инструмента терялся вместе с копией
    ``stream_chat``.
    """
    limit = storage.max_length
    to_store: list[dict[str, Any]] = []
    for item in new_messages:
        cleaned = _sanitize_history_message(item)
        if cleaned is not None:
            to_store.append(cleaned)

    final_text = answer.strip()
    if not final_text and images:
        final_text = _image_completion_text(new_messages, len(images))

    final_index: int | None = None
    for index in range(len(to_store) - 1, -1, -1):
        item = to_store[index]
        if item.get("role") == "assistant" and not item.get("tool_calls"):
            final_index = index
            break

    if final_index is not None:
        if final_text:
            to_store[final_index]["content"] = final_text
        else:
            # Пустой финал без картинок в историю не пишем.
            to_store.pop(final_index)
    elif final_text:
        to_store.append({"role": "assistant", "content": final_text})

    if not to_store:
        return

    for item in to_store:
        await storage.append_message(chat_id, assistant_id, topic_id, item, limit)


async def _active_topic(
    chat_id: int, assistant_id: str, storage: TopicHistoryStorage
) -> str:
    """Возвращает активную тему персоны. На первом сообщении создаёт General."""
    return await storage.ensure_active(chat_id, assistant_id, "General")


async def _reply_history_error(message: Message) -> None:
    logger.exception("Ошибка хранилища истории, chat_id=%s", message.chat.id)
    await message.answer(MSG_HISTORY)


async def _reply_profile_error(message: Message) -> None:
    logger.exception("Ошибка хранилища профилей, chat_id=%s", message.chat.id)
    await message.answer(MSG_PROFILE)


def _public_error(exc: BaseException) -> str:
    """Текст для пользователя. Подробности исключения наружу не выходят."""
    cause = exc.__cause__
    if isinstance(exc, httpx.TimeoutException) or isinstance(
        cause, httpx.TimeoutException
    ):
        return MSG_TIMEOUT
    if isinstance(exc, AssistAIError):
        return MSG_UNAVAILABLE
    return MSG_INTERNAL


def _log_handler_error(exc: BaseException) -> None:
    if isinstance(exc, AssistAIError):
        logger.error("AssistAI ошибка: %s", exc.message)
        return
    if isinstance(exc, httpx.TimeoutException):
        logger.error("Превышено время ожидания внешнего сервиса")
        return
    logger.exception("Непредвиденная ошибка")


async def _send_text(message: Message, text: str) -> None:
    """Отправляет текст частями: у Telegram лимит 4096 символов на сообщение."""
    payload = text.strip() or "Пустой ответ."
    offset = 0
    while offset < len(payload):
        await message.answer(payload[offset : offset + TELEGRAM_TEXT_LIMIT])
        offset += TELEGRAM_TEXT_LIMIT


async def _edit_or_answer(
    query: CallbackQuery, text: str, keyboard: InlineKeyboardMarkup | None = None
) -> None:
    """Редактирует сообщение кнопки. Повтор того же текста не считается ошибкой."""
    message = query.message
    if not isinstance(message, Message):
        return
    try:
        await message.edit_text(text, reply_markup=keyboard)
    except TelegramBadRequest as exc:
        if "message is not modified" in str(exc).lower():
            return
        logger.warning("Не удалось отредактировать сообщение: %s", exc)


async def _edit_stream(status: Message, text: str, previous: str) -> str:
    """Обновляет черновик ответа. Повтор того же текста Telegram отвергает."""
    visible = text[:TELEGRAM_TEXT_LIMIT]
    if not visible or visible == previous:
        return previous
    try:
        await status.edit_text(visible)
    except TelegramBadRequest as exc:
        if "message is not modified" in str(exc).lower():
            return previous
        logger.warning("Не удалось обновить сообщение: %s", exc)
        return previous
    return visible


async def _send_overflow(message: Message, text: str) -> None:
    """Доставляет хвост ответа, который не поместился в редактируемое сообщение."""
    rest = text[TELEGRAM_TEXT_LIMIT:]
    while rest:
        chunk = rest[:TELEGRAM_TEXT_LIMIT]
        rest = rest[TELEGRAM_TEXT_LIMIT:]
        await message.answer(chunk)


def _is_http_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


def _extract_generated_images(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Достаёт картинки из результатов инструмента ``generate_image``.

    Имя функции лежит в ``assistant.tool_calls``, а байты или URL — в
    последующем сообщении ``role: tool`` с тем же ``tool_call_id``.
    Битый JSON пропускается: текстовый ответ модели всё равно уйдёт пользователю.
    """
    names_by_id: dict[str, str] = {}
    for item in messages:
        if item.get("role") != "assistant":
            continue
        tool_calls = item.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            call_id = call.get("id")
            function = call.get("function")
            if not isinstance(call_id, str) or not isinstance(function, dict):
                continue
            name = function.get("name")
            if isinstance(name, str):
                names_by_id[call_id] = name

    images: list[dict[str, Any]] = []
    for item in messages:
        if item.get("role") != "tool":
            continue
        call_id = item.get("tool_call_id")
        if not isinstance(call_id, str) or names_by_id.get(call_id) != "generate_image":
            continue
        content = item.get("content")
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except ValueError:
            logger.warning("Не удалось разобрать результат generate_image")
            continue
        if not isinstance(payload, dict):
            continue
        raw_images = payload.get("images")
        if not isinstance(raw_images, list):
            continue
        for image in raw_images:
            if not isinstance(image, dict):
                continue
            kind = image.get("type")
            value = image.get("value")
            if kind in {"url", "base64"} and isinstance(value, str) and value:
                images.append({"type": kind, "value": value})
    return images


async def _send_extracted_images(
    message: Message, images: list[dict[str, Any]]
) -> None:
    """Отправляет картинки из инструмента. Сбой одной не отменяет остальные."""
    for index, image in enumerate(images):
        value = image.get("value")
        if not isinstance(value, str):
            continue
        try:
            await _send_generated_image(message, value, index)
        except Exception:
            logger.exception("Не удалось отправить изображение из generate_image")


async def _send_outbound_media(message: Message, items: list[OutboundMedia]) -> None:
    """Отправляет файлы из media-tools (видео, аудио, транскрипт, правки)."""
    for item in items:
        try:
            if item.kind == "image":
                if item.url:
                    await message.answer_photo(photo=item.url)
                elif item.data:
                    await message.answer_photo(
                        photo=BufferedInputFile(item.data, filename=item.filename)
                    )
            elif item.kind == "audio":
                if not item.data:
                    continue
                audio = BufferedInputFile(item.data, filename=item.filename)
                if item.filename.endswith((".ogg", ".opus")):
                    await message.answer_voice(voice=audio)
                else:
                    await message.answer_audio(audio=audio)
            elif item.kind == "video":
                if not item.data:
                    continue
                await message.answer_video(
                    video=BufferedInputFile(item.data, filename=item.filename)
                )
            elif item.kind == "document":
                if not item.data:
                    continue
                await message.answer_document(
                    document=BufferedInputFile(item.data, filename=item.filename)
                )
        except Exception:
            logger.exception(
                "outbound_media_send_failed", kind=item.kind, filename=item.filename
            )


async def _send_generated_image(message: Message, image: str, index: int) -> None:
    """URL уходит как ссылка, base64 — как файл. Оба варианта отдаёт generate_image."""
    filename = "image.png" if index == 0 else f"image_{index + 1}.png"
    if _is_http_url(image):
        await message.answer_photo(photo=image)
        return
    try:
        raw = base64.b64decode(image, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise AssistAIError("ProxyAPI вернул нечитаемый base64 изображения.") from exc
    if not raw:
        raise AssistAIError("ProxyAPI вернул пустое изображение.")
    await message.answer_photo(photo=BufferedInputFile(raw, filename=filename))


def filter_sensitive_data(
    event: dict[str, Any], hint: dict[str, Any]
) -> dict[str, Any]:
    """Strip secrets from Sentry payloads before upload."""
    _ = hint
    request = event.get("request")
    if isinstance(request, dict):
        data = request.get("data")
        if isinstance(data, dict):
            for key in ("password", "token", "api_key", "Authorization"):
                if key in data:
                    data[key] = "[REDACTED]"
    return event


def init_sentry() -> None:
    """Initialize Sentry when ``SENTRY_DSN`` is present."""
    dsn = os.getenv("SENTRY_DSN", "").strip()
    if not dsn:
        logger.warning("sentry_disabled", reason="SENTRY_DSN not set")
        return
    sentry_sdk.init(
        dsn=dsn,
        environment=os.getenv("ENVIRONMENT", "development"),
        release=os.getenv("APP_VERSION", "1.0.0"),
        integrations=[
            AioHttpIntegration(),
            LoggingIntegration(level=logging.INFO, event_level=logging.ERROR),
        ],
        traces_sample_rate=0.1,
        send_default_pii=False,
        before_send=filter_sensitive_data,
    )
    logger.info(
        "sentry_initialized",
        environment=os.getenv("ENVIRONMENT", "development"),
    )


async def create_fsm_storage(redis_url: str) -> BaseStorage:
    """Redis FSM storage with MemoryStorage fallback."""
    if not redis_url:
        logger.warning("fsm_memory_storage", reason="REDIS_URL not set")
        return MemoryStorage()
    try:
        from aiogram.fsm.storage.redis import RedisStorage

        probe = Redis.from_url(
            redis_url,
            socket_connect_timeout=2.0,
            socket_timeout=2.0,
        )
        try:
            await probe.ping()
        finally:
            await probe.aclose()
        storage = RedisStorage.from_url(redis_url)
        logger.info("fsm_redis_storage", url=redis_url)
        return storage
    except Exception as exc:
        logger.error("fsm_redis_failed", error=str(exc), fallback="memory")
        return MemoryStorage()


async def _open_shared_redis(redis_url: str) -> Redis | None:
    """Optional Redis client for rate limits and health checks."""
    if not redis_url:
        return None
    try:
        client = Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=2.0,
            socket_timeout=2.0,
        )
        await client.ping()
        return client
    except Exception as exc:
        logger.warning("shared_redis_unavailable", error=str(exc))
        return None


async def on_startup(dispatcher: Dispatcher, bot: Bot) -> None:
    """Создаёт клиент LLM, темы и профили ассистентов. Все живут в workflow_data."""
    topic_storage: TopicHistoryStorage | None = None
    assistant_storage: AssistantStorage | None = None
    redis_url = os.getenv("REDIS_URL", "").strip()
    shared_redis = await _open_shared_redis(redis_url)
    rate_limiter.attach_redis(shared_redis)
    cost_tracker.attach_redis(shared_redis)
    await job_queue.init_job_queue(redis_url)
    dispatcher.workflow_data["shared_redis"] = shared_redis
    try:
        topic_storage = await create_topic_storage(TopicConfig.from_env())
        assistant_storage = await create_assistant_storage(RedisConfig.from_env())
        assistant = AssistAIClient(AssistAIConfig.from_env())
    except ValidationError:
        logger.exception("bot_config_invalid")
        if assistant_storage is not None:
            await assistant_storage.close()
        if topic_storage is not None:
            await topic_storage.close()
        if shared_redis is not None:
            await shared_redis.aclose()
        raise
    dispatcher.workflow_data["topic_storage"] = topic_storage
    dispatcher.workflow_data["assistant_storage"] = assistant_storage
    dispatcher.workflow_data["assistant"] = assistant

    metrics_port = int(os.getenv("METRICS_PORT", "8083"))
    health_port = int(os.getenv("HEALTH_PORT", "8082"))
    start_metrics_server(port=metrics_port)

    upload_store = UploadSessionStore(shared_redis)
    s3_storage = get_storage()
    assistant.upload_session_store = upload_store  # type: ignore[attr-defined]
    dispatcher.workflow_data["bot"] = bot
    dispatcher.workflow_data["upload_store"] = upload_store
    dispatcher.workflow_data["s3_storage"] = s3_storage

    def _setup_upload_http(app: Any) -> None:
        if s3_storage is None or not os.getenv("UPLOAD_PUBLIC_BASE_URL", "").strip():
            logger.info(
                "large_upload_disabled",
                reason="YC_* or UPLOAD_PUBLIC_BASE_URL not configured",
            )
            return

        async def _on_complete(session: UploadSession) -> None:
            await _process_s3_upload_session(
                bot=bot,
                assistant=assistant,
                topic_storage=topic_storage,
                assistant_storage=assistant_storage,
                store=upload_store,
                storage=s3_storage,
                session=session,
            )

        setup_upload_routes(
            app,
            store=upload_store,
            storage=s3_storage,
            on_complete=_on_complete,
        )
        app["upload_enabled"] = True
        logger.info(
            "large_upload_enabled",
            public_base=os.getenv("UPLOAD_PUBLIC_BASE_URL", "").strip(),
        )

    health_runner = await start_health_server(
        port=health_port,
        redis_client=shared_redis,
        proxy_probe=assistant.test_connection,
        setup_app=_setup_upload_http,
    )
    dispatcher.workflow_data["health_runner"] = health_runner
    logger.info(
        "assistant_ready",
        tools=len(assistant.get_tools_schema()),
        metrics_port=metrics_port,
        health_port=health_port,
        large_upload=is_upload_feature_ready(),
    )
    try:
        await bot.set_my_commands(
            [
                BotCommand(command="help", description="Что умеет бот"),
            ]
        )
        await bot.set_chat_menu_button(menu_button=MenuButtonCommands())
    except Exception:
        logger.exception("bot_commands_failed")


async def on_shutdown(dispatcher: Dispatcher) -> None:
    """Закрывает клиент LLM, темы и профили, если startup успел их создать."""
    health_runner = dispatcher.workflow_data.get("health_runner")
    if health_runner is not None:
        await health_runner.cleanup()
    await job_queue.close_job_queue()
    assistant = dispatcher.workflow_data.get("assistant")
    if isinstance(assistant, AssistAIClient):
        await assistant.close()
        logger.info("assistant_closed")
    storage = dispatcher.workflow_data.get("topic_storage")
    if isinstance(storage, TopicHistoryStorage):
        await storage.close()
    profiles = dispatcher.workflow_data.get("assistant_storage")
    if isinstance(profiles, AssistantStorage):
        await profiles.close()
    shared_redis = dispatcher.workflow_data.get("shared_redis")
    if isinstance(shared_redis, Redis):
        await shared_redis.aclose()
    fsm_storage = dispatcher.storage
    close = getattr(fsm_storage, "close", None)
    if callable(close):
        result = close()
        if asyncio.iscoroutine(result):
            await result


async def cmd_start(
    message: Message,
    state: FSMContext,
    assistant_storage: AssistantStorage,
    topic_storage: TopicHistoryStorage,
) -> None:
    """Приветствие, список возможностей и постоянное меню."""
    await _reset_flow(state)
    menu = await _menu_for(
        message.chat.id,
        assistant_storage,
        topic_storage,
        ensure_topic=True,
    )
    await message.answer(WELCOME, reply_markup=menu)


async def cmd_cancel(
    message: Message,
    state: FSMContext,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    topic_storage: TopicHistoryStorage,
) -> None:
    """Сбрасывает текущий пошаговый ввод. Без шага отвечает, что отменять нечего."""
    menu = await _menu_for(message.chat.id, assistant_storage, topic_storage)
    current = await state.get_state()
    if current is None:
        await _apply_reply_menu(message, menu, state)
        await message.answer("Нечего отменять.")
        return
    return_to_settings = current in {
        EditAssistantSG.name.state,
        EditAssistantSG.prompt.state,
        NewAssistantSG.name.state,
        NewAssistantSG.prompt.state,
        NewAssistantSG.models.state,
    }
    return_to_topics = current in {
        RenameTopicSG.name.state,
        NewTopicSG.name.state,
    }
    await _reset_flow(state)
    await _apply_reply_menu(message, menu, state)
    if return_to_settings:
        await _open_settings(message, assistant, assistant_storage, state)
    elif return_to_topics:
        await _open_topics(message, topic_storage, assistant_storage, state)


async def cmd_unknown(message: Message) -> None:
    """Отвечает на команду, для которой нет отдельного хендлера."""
    await message.answer("Команды убраны. Пользуйтесь кнопками внизу или /help.")


def _should_send_voice_reply(profile: AssistantProfile, input_kind: InputKind) -> bool:
    """Нужен ли голосовой ответ поверх текста."""
    if profile.voice_mode == "off":
        return False
    if profile.voice_mode == "always":
        return True
    return profile.voice_mode == "auto" and input_kind == "voice"


def _clip_tts_text(text: str) -> str:
    """Обрезает длинный ответ для TTS по последнему пробелу."""
    cleaned = text.strip()
    if len(cleaned) <= TTS_MAX_CHARS:
        return cleaned
    cut = cleaned[:TTS_MAX_CHARS]
    space = cut.rfind(" ")
    if space > TTS_MAX_CHARS // 2:
        cut = cut[:space]
    logger.warning(
        "TTS обрезан: было %s символов, озвучено %s",
        len(cleaned),
        len(cut),
    )
    return cut.rstrip()


async def _send_voice_reply(
    message: Message,
    assistant: AssistAIClient,
    profile: AssistantProfile,
    text: str,
) -> None:
    """Озвучивает ответ персоны. Сбой TTS не откатывает текст и историю."""
    tts_text = _clip_tts_text(text)
    if not tts_text:
        return
    model = profile.tts_model or assistant.config.default_tts_model
    voice = profile.tts_voice or assistant.config.default_tts_voice
    fmt = profile.tts_format or assistant.config.default_tts_format
    user_id = message.from_user.id if message.from_user else message.chat.id
    set_billing_user(user_id)
    try:
        await cost_tracker.check_budget(
            user_id, cost_tracker.estimate_tts_cost(tts_text)
        )
        speech = await job_queue.synthesize_speech(
            assistant,
            tts_text,
            model=model,
            voice=voice,
            response_format=fmt,
        )
        await cost_tracker.record(
            kind="tts",
            amount_usd=cost_tracker.estimate_tts_cost(tts_text),
            user_id=user_id,
        )
    except BudgetExceededError as exc:
        await message.answer(exc.message)
        return
    except (AssistAIError, httpx.TimeoutException, ValueError) as exc:
        _log_handler_error(exc)
        await message.answer("⚠️ Не удалось озвучить ответ.")
        return
    except Exception:
        logger.exception("Непредвиденная ошибка TTS")
        await message.answer("⚠️ Не удалось озвучить ответ.")
        return
    audio = BufferedInputFile(speech.audio, filename=speech.filename)
    try:
        await message.answer_voice(voice=audio)
    except TelegramBadRequest as exc:
        logger.warning("Telegram не принял voice, пробую audio: %s", exc)
        try:
            await message.answer_audio(
                audio=BufferedInputFile(speech.audio, filename=speech.filename)
            )
        except TelegramBadRequest as nested:
            logger.warning("Не удалось отправить озвучку: %s", nested)


async def _process_user_message(
    message: Message,
    assistant: AssistAIClient,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    state: FSMContext,
    user_text: str,
    input_kind: InputKind = "text",
    status: Message | None = None,
    user_media: list[UserMedia] | None = None,
) -> None:
    """Общий pipeline текста и распознанного голоса в активной теме."""
    chat_id = message.chat.id
    user_id = message.from_user.id if message.from_user else chat_id
    started = time.perf_counter()
    bind_request_context(user_id=user_id, chat_id=chat_id)
    set_billing_user(user_id)
    note_active_user(user_id)
    sentry_sdk.set_user({"id": str(user_id)})
    sentry_sdk.set_context("chat", {"id": chat_id, "type": message.chat.type})
    media_token = set_user_media(user_media or [])
    upload_ctx_token = set_upload_request_context(
        UploadRequestContext(chat_id=chat_id, user_id=user_id)
    )
    try:
        await cost_tracker.check_budget(
            user_id,
            cost_tracker.estimate_chat_cost(
                input_chars=len(user_text), output_chars=500
            ),
        )
    except BudgetExceededError as exc:
        await message.answer(exc.message)
        reset_user_media(media_token)
        reset_upload_request_context(upload_ctx_token)
        clear_request_context()
        return
    try:
        profile = await _active_profile(chat_id, assistant_storage)
        topic_id = await _active_topic(chat_id, profile.id, topic_storage)
        bind_request_context(
            user_id=user_id,
            chat_id=chat_id,
            assistant_id=profile.id,
            topic_id=topic_id,
        )
        history_copy = await _prepare_messages(
            chat_id, profile.id, user_text, topic_id, topic_storage, profile
        )
    except AssistantStorageError:
        reset_upload_request_context(upload_ctx_token)
        await _reply_profile_error(message)
        return
    except TopicStorageError:
        reset_upload_request_context(upload_ctx_token)
        await _reply_history_error(message)
        return
    text_model = _text_model_of(profile, assistant)
    logger.info(
        "processing_message",
        kind=input_kind,
        model=text_model,
        message_count=len(history_copy),
    )
    keep: Token[bool] | None = _KEEP_OUTGOING.set(True)
    if status is None:
        status = await message.answer("Готовлю ответ…")
    parts: list[str] = []
    shown = ""
    last_edit = 0.0
    scope = ImageModelScope(profile.image_model, profile.image_size)
    media_scope = MediaTurnScope()
    turn_start = len(history_copy)
    try:
        if message.bot is not None:
            await message.bot.send_chat_action(chat_id=chat_id, action="typing")
        with scope, media_scope:
            async for delta in assistant.stream_chat(
                messages=history_copy, model=text_model
            ):
                if delta.kind != "content" or not delta.text:
                    continue
                parts.append(delta.text)
                now = asyncio.get_running_loop().time()
                if now - last_edit < STREAM_EDIT_INTERVAL:
                    continue
                shown = await _edit_stream(status, "".join(parts), shown)
                last_edit = now
        answer = "".join(parts).strip()
        images = scope.images
        outbound = media_scope.bucket.items
        if not answer and not images and not outbound:
            await _rollback_user(chat_id, profile.id, topic_storage, topic_id, message)
            await _edit_stream(status, "Пустой ответ.", shown)
            keep = _release_outgoing(keep)
            await _append_chrome(state, [status.message_id])
            return
        if answer:
            visible = await _edit_stream(status, answer, shown)
            head = answer[:TELEGRAM_TEXT_LIMIT]
            if visible == head:
                await _send_overflow(message, answer)
            else:
                await _send_text(message, answer)
        else:
            await _delete_bot_message(status)
        keep = _release_outgoing(keep)
        try:
            await _remember_model_turn(
                chat_id,
                profile.id,
                topic_id,
                topic_storage,
                history_copy[turn_start:],
                answer=answer,
                images=images,
            )
        except TopicStorageError:
            fallback = answer.strip()
            if not fallback and images:
                fallback = _image_completion_text(
                    history_copy[turn_start:], len(images)
                )
            if fallback:
                try:
                    await topic_storage.append_message(
                        chat_id,
                        profile.id,
                        topic_id,
                        {"role": "assistant", "content": fallback},
                        topic_storage.max_length,
                    )
                except TopicStorageError:
                    await _reply_history_error(message)
            else:
                await _reply_history_error(message)
        await _send_extracted_images(message, images)
        await _send_outbound_media(message, outbound)
        logger.info(
            "message_answered",
            chars=len(answer),
            images=len(images),
            outbound=len(outbound),
        )
        if answer and _should_send_voice_reply(profile, input_kind):
            await _send_voice_reply(message, assistant, profile, answer)
    except (AssistAIError, httpx.TimeoutException) as exc:
        keep = _release_outgoing(keep)
        await _send_extracted_images(message, scope.images)
        await _send_outbound_media(message, media_scope.bucket.items)
        _log_handler_error(exc)
        sentry_sdk.capture_exception(exc)
        await _rollback_user(chat_id, profile.id, topic_storage, topic_id, message)
        await _report_stream_error(message, status, parts, _public_error(exc), state)
    except Exception as exc:
        keep = _release_outgoing(keep)
        await _send_extracted_images(message, scope.images)
        await _send_outbound_media(message, media_scope.bucket.items)
        _log_handler_error(exc)
        sentry_sdk.capture_exception(exc)
        await _rollback_user(chat_id, profile.id, topic_storage, topic_id, message)
        await _report_stream_error(message, status, parts, MSG_INTERNAL, state)
    finally:
        reset_user_media(media_token)
        reset_upload_request_context(upload_ctx_token)
        observe_message(input_kind, time.perf_counter() - started)
        clear_request_context()
        _release_outgoing(keep)


async def handle_text(
    message: Message,
    assistant: AssistAIClient,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Обычный диалог в активной теме. Ответ языковой модели идёт потоком."""
    user_text = (message.text or "").strip()
    if not user_text:
        return
    user_id = message.from_user.id if message.from_user else message.chat.id
    if not security.validate_message_length(user_text):
        await message.answer("⚠️ Сообщение слишком длинное. Максимум 4000 символов.")
        return
    if not await rate_limiter.check_text_limit(user_id):
        retry_after = await rate_limiter.get_retry_after(user_id, "text")
        await message.answer(
            f"⏳ Слишком много запросов. Подождите {int(retry_after) or 60} секунд."
        )
        logger.warning("rate_limit_exceeded", user_id=user_id, operation="text")
        return
    sanitized_text = security.sanitize_prompt(user_text)
    await _process_user_message(
        message,
        assistant,
        topic_storage,
        assistant_storage,
        state,
        user_text=sanitized_text,
        input_kind="text",
    )


async def _download_voice(message: Message) -> tuple[bytes, str]:
    """Скачивает голосовое сообщение Telegram в память.

    Returns:
        ``(audio_bytes, filename)``.

    Raises:
        ValueError: нет voice или пустые байты.
        AssistAIError: сбой скачивания.
    """
    voice = message.voice
    if voice is None:
        raise ValueError("В сообщении нет голосового вложения.")
    bot = message.bot
    if bot is None:
        raise AssistAIError("Бот недоступен для скачивания голосового.")
    buffer = io.BytesIO()
    try:
        await bot.download(voice, destination=buffer)
    except Exception as exc:
        logger.exception("Не удалось скачать голосовое")
        raise AssistAIError("Не удалось скачать голосовое сообщение.") from exc
    audio_bytes = buffer.getvalue()
    if not audio_bytes:
        raise ValueError("Голосовое сообщение пустое.")
    return audio_bytes, "voice.ogg"


async def handle_voice(
    message: Message,
    assistant: AssistAIClient,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Распознаёт голосовое и передаёт текст в общий pipeline диалога."""
    voice = message.voice
    if voice is None:
        return
    if voice.duration > MAX_VOICE_DURATION_SECONDS:
        await message.answer("Голосовое слишком длинное. Максимум — 10 минут.")
        return
    if (
        voice.file_size is not None
        and voice.file_size > MAX_VOICE_SIZE_MB * 1024 * 1024
    ):
        await message.answer("Голосовое слишком большое. Максимум — 20 МБ.")
        return
    chat_id = message.chat.id
    user_id = message.from_user.id if message.from_user else chat_id
    if not await rate_limiter.check_text_limit(user_id):
        retry_after = await rate_limiter.get_retry_after(user_id, "text")
        await message.answer(
            f"⏳ Слишком много запросов. Подождите {int(retry_after) or 60} секунд."
        )
        logger.warning("rate_limit_exceeded", user_id=user_id, operation="text")
        return
    logger.info(
        "voice_received",
        duration=voice.duration,
        size=voice.file_size,
        user_id=user_id,
        chat_id=chat_id,
    )
    set_billing_user(user_id)
    keep: Token[bool] | None = _KEEP_OUTGOING.set(True)
    status = await message.answer("🎤 Распознаю речь…")
    keep = _release_outgoing(keep)
    try:
        await cost_tracker.check_budget(
            user_id, cost_tracker.estimate_stt_cost(float(voice.duration))
        )
        audio_bytes, filename = await _download_voice(message)
        profile = await _active_profile(chat_id, assistant_storage)
        transcribed = await job_queue.transcribe_audio(
            assistant,
            audio_bytes,
            filename=filename,
            language=profile.stt_language,
        )
        await cost_tracker.record(
            kind="stt",
            amount_usd=cost_tracker.estimate_stt_cost(float(voice.duration)),
            user_id=user_id,
        )
    except BudgetExceededError as exc:
        await _edit_stream(status, exc.message, "")
        await _append_chrome(state, [status.message_id])
        return
    except AssistantStorageError:
        await _edit_stream(status, MSG_INTERNAL, "")
        await _append_chrome(state, [status.message_id])
        await _reply_profile_error(message)
        return
    except ValueError as exc:
        await _edit_stream(status, str(exc) or "Не удалось обработать голосовое.", "")
        await _append_chrome(state, [status.message_id])
        return
    except AssistAIError as exc:
        logger.error("AssistAI ошибка STT: %s", exc.message)
        await _edit_stream(
            status, "⚠️ Не удалось распознать речь. Попробуйте ещё раз.", ""
        )
        await _append_chrome(state, [status.message_id])
        return
    except httpx.TimeoutException:
        await _edit_stream(status, "⚠️ Превышено время ожидания распознавания.", "")
        await _append_chrome(state, [status.message_id])
        return
    except Exception:
        logger.exception("Непредвиденная ошибка STT")
        await _edit_stream(status, MSG_INTERNAL, "")
        await _append_chrome(state, [status.message_id])
        return
    logger.info(
        "Распознано, chat_id=%s duration=%s size=%s chars=%s",
        chat_id,
        voice.duration,
        voice.file_size,
        len(transcribed),
    )
    if not transcribed:
        await _edit_stream(status, "🤷 В голосовом не распознано речи.", "")
        await _append_chrome(state, [status.message_id])
        return
    if SHOW_RECOGNIZED_TEXT:
        preview = transcribed[:VOICE_RECOGNIZED_PREVIEW_LIMIT]
        await _edit_stream(
            status,
            f"🗣 Вы сказали:\n«{preview}»\n\nГотовлю ответ…",
            "",
        )
    await _process_user_message(
        message,
        assistant,
        topic_storage,
        assistant_storage,
        state,
        user_text=transcribed,
        input_kind="voice",
        status=status,
    )


def _infer_upload_media_kind(*, mime_type: str, filename: str, preferred: str) -> str:
    if preferred in {"audio", "video"}:
        return preferred
    mime = (mime_type or "").lower()
    name = (filename or "").lower()
    if mime.startswith("video/") or name.endswith((".mp4", ".mov", ".mkv")):
        return "video"
    if name.endswith(".webm") and not mime.startswith("audio/"):
        return "video"
    return "audio"


async def _process_s3_upload_session(
    *,
    bot: Bot,
    assistant: AssistAIClient,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    store: UploadSessionStore,
    storage: S3Storage,
    session: UploadSession,
) -> None:
    """Скачивает объект из бакета и делает STT → transcript.txt в чат."""
    chat_id = session.chat_id
    object_key = session.object_key
    filename = session.filename or "media.bin"
    mime_type = session.mime_type or "application/octet-stream"
    if not object_key:
        await bot.send_message(
            chat_id, "Загрузка без object key — нечего обрабатывать."
        )
        session.status = "failed"
        session.error = "missing_object_key"
        await store.update(session)
        return

    status_msg = await bot.send_message(
        chat_id, f"Файл «{filename}» получен. Расшифровываю…"
    )
    data: bytes | None = None
    try:
        data = await storage.download_bytes(object_key)
        kind = _infer_upload_media_kind(
            mime_type=mime_type,
            filename=filename,
            preferred=session.media_kind,
        )
        chunks = await prepare_audio_for_stt(
            data,
            filename=filename,
            is_video=kind == "video",
        )
        parts: list[str] = []
        for audio_bytes, chunk_name in chunks:
            text = await job_queue.transcribe_audio(
                assistant,
                audio_bytes,
                filename=chunk_name,
            )
            if text:
                parts.append(text)
        transcript = "\n\n".join(parts).strip()
        if not transcript:
            await bot.send_message(chat_id, "Речь не распознана.")
            session.status = "done"
            await store.update(session)
            return

        await bot.send_document(
            chat_id,
            document=BufferedInputFile(
                transcript.encode("utf-8"), filename="transcript.txt"
            ),
            caption=f"Расшифровка «{filename}» ({len(transcript)} симв.)",
        )
        try:
            await status_msg.delete()
        except Exception:
            logger.debug("upload_status_delete_failed", chat_id=chat_id)

        # Короткая запись в историю активной темы.
        try:
            profile = await _active_profile(chat_id, assistant_storage)
            topic_id = await _active_topic(chat_id, profile.id, topic_storage)
            await topic_storage.append_message(
                chat_id,
                profile.id,
                topic_id,
                {
                    "role": "user",
                    "content": f"[Большой файл загружен в облако: {filename}]",
                },
                topic_storage.max_length,
            )
            await topic_storage.append_message(
                chat_id,
                profile.id,
                topic_id,
                {
                    "role": "assistant",
                    "content": (
                        f"Расшифровка «{filename}» отправлена файлом transcript.txt "
                        f"({len(transcript)} символов)."
                    ),
                },
                topic_storage.max_length,
            )
        except Exception:
            logger.exception("upload_history_append_failed", chat_id=chat_id)

        session.status = "done"
        await store.update(session)
        logger.info(
            "s3_upload_transcribed",
            upload_id=session.upload_id,
            chat_id=chat_id,
            chars=len(transcript),
            chunks=len(chunks),
        )
    except (S3StorageError, MediaProcessingError, AssistAIError) as exc:
        message = getattr(exc, "message", None) or str(exc)
        logger.exception("s3_upload_process_failed", upload_id=session.upload_id)
        await bot.send_message(chat_id, f"Не удалось обработать файл: {message}")
        session.status = "failed"
        session.error = message[:500]
        await store.update(session)
    except Exception as exc:
        logger.exception("s3_upload_process_failed", upload_id=session.upload_id)
        sentry_sdk.capture_exception(exc)
        await bot.send_message(chat_id, "Не удалось обработать загруженный файл.")
        session.status = "failed"
        session.error = "internal"
        await store.update(session)
    finally:
        data = None
        try:
            await storage.delete_object(object_key)
        except Exception:
            logger.warning("s3_cleanup_failed", key=object_key)


async def _process_attached_media(
    message: Message,
    assistant: AssistAIClient,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    state: FSMContext,
    *,
    data: bytes,
    filename: str,
    mime_type: str,
    kind: str,
) -> None:
    """Передаёт аудио/видео персоне: tools сами решают (STT и т.д.)."""
    caption = (message.caption or "").strip()
    if kind == "audio":
        placeholder = f"[Аудиофайл: {filename}]"
        default_ask = (
            "Пользователь прислал аудиофайл. Если нужна расшифровка — "
            "вызови transcribe_media."
        )
        media = [
            UserMedia(kind="audio", filename=filename, mime_type=mime_type, data=data)
        ]
    else:
        placeholder = f"[Видеофайл: {filename}]"
        default_ask = (
            "Пользователь прислал видео. Если нужна расшифровка речи — "
            "вызови transcribe_media (аудио извлекается автоматически)."
        )
        media = [
            UserMedia(kind="video", filename=filename, mime_type=mime_type, data=data)
        ]
    user_text = caption if caption else f"{placeholder}\n{default_ask}"
    status = await message.answer("Обрабатываю файл…")
    await _process_user_message(
        message,
        assistant,
        topic_storage,
        assistant_storage,
        state,
        user_text=user_text,
        input_kind="text",
        status=status,
        user_media=media,
    )


async def handle_audio(
    message: Message,
    assistant: AssistAIClient,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Аудиофайл → персона (не авто-STT; расшифровка через tool)."""
    audio = message.audio
    if audio is None:
        return
    if await state.get_state() is not None:
        return
    user_id = message.from_user.id if message.from_user else message.chat.id
    if not await rate_limiter.check_text_limit(user_id):
        retry_after = await rate_limiter.get_retry_after(user_id, "text")
        await message.answer(
            f"⏳ Слишком много запросов. Подождите {int(retry_after) or 60} секунд."
        )
        return
    if audio.file_size and audio.file_size > MAX_MEDIA_SIZE_MB * 1024 * 1024:
        await message.answer(
            f"Файл слишком большой. Лимит Telegram Bot API — {MAX_MEDIA_SIZE_MB} МБ."
        )
        return
    bot = message.bot
    if bot is None:
        await message.answer("Не удалось загрузить файл")
        return
    try:
        buffer = io.BytesIO()
        await bot.download(audio, destination=buffer)
        data = buffer.getvalue()
    except Exception:
        logger.exception("audio_download_failed")
        await message.answer("Не удалось скачать аудио.")
        return
    if not data:
        await message.answer("Пустой аудиофайл.")
        return
    filename = audio.file_name or "audio.mp3"
    mime = audio.mime_type or "audio/mpeg"
    await _process_attached_media(
        message,
        assistant,
        topic_storage,
        assistant_storage,
        state,
        data=data,
        filename=filename,
        mime_type=mime,
        kind="audio",
    )


async def handle_video_file(
    message: Message,
    assistant: AssistAIClient,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Видео / video_note → персона; STT через tool после извлечения аудио."""
    video = message.video or message.video_note
    if video is None:
        return
    if await state.get_state() is not None:
        return
    user_id = message.from_user.id if message.from_user else message.chat.id
    if not await rate_limiter.check_text_limit(user_id):
        retry_after = await rate_limiter.get_retry_after(user_id, "text")
        await message.answer(
            f"⏳ Слишком много запросов. Подождите {int(retry_after) or 60} секунд."
        )
        return
    size = getattr(video, "file_size", None)
    if size and size > MAX_MEDIA_SIZE_MB * 1024 * 1024:
        await message.answer(
            f"Файл слишком большой. Лимит Telegram Bot API — {MAX_MEDIA_SIZE_MB} МБ."
        )
        return
    bot = message.bot
    if bot is None:
        await message.answer("Не удалось загрузить файл")
        return
    try:
        buffer = io.BytesIO()
        await bot.download(video, destination=buffer)
        data = buffer.getvalue()
    except Exception:
        logger.exception("video_download_failed")
        await message.answer("Не удалось скачать видео.")
        return
    if not data:
        await message.answer("Пустой видеофайл.")
        return
    filename = getattr(video, "file_name", None) or "video.mp4"
    mime = getattr(video, "mime_type", None) or "video/mp4"
    await _process_attached_media(
        message,
        assistant,
        topic_storage,
        assistant_storage,
        state,
        data=data,
        filename=filename,
        mime_type=mime,
        kind="video",
    )


async def _process_vision_message(
    message: Message,
    assistant: AssistAIClient,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    state: FSMContext,
    *,
    multimodal_content: list[dict[str, Any]],
    history_placeholder: str,
    status_text: str = "Анализирую изображение…",
    user_media: list[UserMedia] | None = None,
) -> None:
    """Стримит ответ модели по multimodal-запросу; в тему пишет только текст."""
    chat_id = message.chat.id
    media_token = set_user_media(user_media or [])
    try:
        profile = await _active_profile(chat_id, assistant_storage)
        topic_id = await _active_topic(chat_id, profile.id, topic_storage)
        history_copy = await _prepare_messages_with_image(
            chat_id,
            profile.id,
            multimodal_content,
            history_placeholder,
            topic_id,
            topic_storage,
            profile,
        )
    except AssistantStorageError:
        reset_user_media(media_token)
        await _reply_profile_error(message)
        return
    except TopicStorageError:
        reset_user_media(media_token)
        await _reply_history_error(message)
        return
    vision_model = _vision_model_of(profile, assistant)
    logger.info(
        "Vision, chat_id=%s topic_id=%s, модель=%s, сообщений в запросе=%s",
        chat_id,
        topic_id,
        vision_model,
        len(history_copy),
    )
    keep: Token[bool] | None = _KEEP_OUTGOING.set(True)
    status = await message.answer(status_text)
    parts: list[str] = []
    shown = ""
    last_edit = 0.0
    scope = ImageModelScope(profile.image_model, profile.image_size)
    media_scope = MediaTurnScope()
    turn_start = len(history_copy)
    try:
        if message.bot is not None:
            await message.bot.send_chat_action(chat_id=chat_id, action="typing")
        with scope, media_scope:
            async for delta in assistant.stream_chat(
                messages=history_copy, model=vision_model
            ):
                if delta.kind != "content" or not delta.text:
                    continue
                parts.append(delta.text)
                now = asyncio.get_running_loop().time()
                if now - last_edit < STREAM_EDIT_INTERVAL:
                    continue
                shown = await _edit_stream(status, "".join(parts), shown)
                last_edit = now
        answer = "".join(parts).strip()
        images = scope.images
        outbound = media_scope.bucket.items
        if not answer and not images and not outbound:
            await _rollback_user(chat_id, profile.id, topic_storage, topic_id, message)
            await _edit_stream(status, "Пустой ответ.", shown)
            keep = _release_outgoing(keep)
            await _append_chrome(state, [status.message_id])
            return
        if answer:
            visible = await _edit_stream(status, answer, shown)
            head = answer[:TELEGRAM_TEXT_LIMIT]
            if visible == head:
                await _send_overflow(message, answer)
            else:
                await _send_text(message, answer)
        else:
            await _delete_bot_message(status)
        keep = _release_outgoing(keep)
        try:
            await _remember_model_turn(
                chat_id,
                profile.id,
                topic_id,
                topic_storage,
                history_copy[turn_start:],
                answer=answer,
                images=images,
            )
        except TopicStorageError:
            fallback = answer.strip()
            if not fallback and images:
                fallback = _image_completion_text(
                    history_copy[turn_start:], len(images)
                )
            if fallback:
                try:
                    await topic_storage.append_message(
                        chat_id,
                        profile.id,
                        topic_id,
                        {"role": "assistant", "content": fallback},
                        topic_storage.max_length,
                    )
                except TopicStorageError:
                    await _reply_history_error(message)
            else:
                await _reply_history_error(message)
        await _send_extracted_images(message, images)
        await _send_outbound_media(message, outbound)
        logger.info(
            "Vision-ответ, chat_id=%s topic_id=%s, символов=%s, картинок=%s",
            chat_id,
            topic_id,
            len(answer),
            len(images),
        )
        if answer and _should_send_voice_reply(profile, "text"):
            await _send_voice_reply(message, assistant, profile, answer)
    except (AssistAIError, httpx.TimeoutException) as exc:
        keep = _release_outgoing(keep)
        await _send_extracted_images(message, scope.images)
        await _send_outbound_media(message, media_scope.bucket.items)
        _log_handler_error(exc)
        await _rollback_user(chat_id, profile.id, topic_storage, topic_id, message)
        await _report_stream_error(message, status, parts, _public_error(exc), state)
    except Exception as exc:
        keep = _release_outgoing(keep)
        await _send_extracted_images(message, scope.images)
        await _send_outbound_media(message, media_scope.bucket.items)
        _log_handler_error(exc)
        await _rollback_user(chat_id, profile.id, topic_storage, topic_id, message)
        await _report_stream_error(message, status, parts, MSG_UNAVAILABLE, state)
    finally:
        reset_user_media(media_token)
        _release_outgoing(keep)


async def handle_photo(
    message: Message,
    assistant: AssistAIClient,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Анализирует фото через multimodal stream_chat."""
    if not message.photo:
        return
    if await state.get_state() is not None:
        return
    user_id = message.from_user.id if message.from_user else message.chat.id
    if not await rate_limiter.check_text_limit(user_id):
        retry_after = await rate_limiter.get_retry_after(user_id, "text")
        await message.answer(
            f"⏳ Слишком много запросов. Подождите {int(retry_after) or 60} секунд."
        )
        logger.warning("rate_limit_exceeded", user_id=user_id, operation="text")
        return
    bot = message.bot
    if bot is None:
        await message.answer("Не удалось загрузить файл")
        return
    largest = message.photo[-1]
    if (
        largest.file_size is not None
        and largest.file_size > MAX_IMAGE_SIZE_MB * 1024 * 1024
    ):
        await message.answer(
            f"Файл слишком большой. Максимум — {MAX_IMAGE_SIZE_MB} МБ."
        )
        return
    try:
        data = await download_telegram_file(bot, largest.file_id)
        ensure_within_size_limit(data, label="Фото")
        logger.info(
            "Фото, chat_id=%s size=%s",
            message.chat.id,
            len(data),
        )
        image_part = build_image_content(data, "image/jpeg")
        caption = message.caption or ""
        multimodal = build_multimodal_content(caption, [image_part])
        placeholder = f"[Изображение: {caption.strip() or 'без подписи'}]"
        user_media = [
            UserMedia(
                kind="image",
                filename="photo.jpg",
                mime_type="image/jpeg",
                data=data,
            )
        ]
    except VisionError as exc:
        await message.answer(exc.message)
        return
    await _process_vision_message(
        message,
        assistant,
        topic_storage,
        assistant_storage,
        state,
        multimodal_content=multimodal,
        history_placeholder=placeholder,
        user_media=user_media,
    )


def _document_media_kind(mime: str, file_name: str) -> Literal["audio", "video"] | None:
    """Определяет аудио/видео, присланные как document (файл без сжатия)."""
    mime_l = mime.strip().casefold()
    name_l = file_name.strip().casefold()
    video_ext = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".3gp")
    audio_ext = (".mp3", ".wav", ".m4a", ".ogg", ".oga", ".flac", ".aac", ".opus")
    if mime_l.startswith("video/") or name_l.endswith(video_ext):
        return "video"
    if mime_l.startswith("audio/") or name_l.endswith(audio_ext):
        return "audio"
    return None


async def handle_document(
    message: Message,
    assistant: AssistAIClient,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Документ: картинка/PDF через Vision; аудио/видео — в media pipeline персоны."""
    if await state.get_state() is not None:
        return
    user_id = message.from_user.id if message.from_user else message.chat.id
    if not await rate_limiter.check_text_limit(user_id):
        retry_after = await rate_limiter.get_retry_after(user_id, "text")
        await message.answer(
            f"⏳ Слишком много запросов. Подождите {int(retry_after) or 60} секунд."
        )
        logger.warning("rate_limit_exceeded", user_id=user_id, operation="text")
        return

    document = message.document
    bot = message.bot
    if bot is None:
        await message.answer("Не удалось загрузить файл")
        return

    # Видео/аудио, отправленные «как файл», приходят document, а не video/audio.
    if document is not None:
        mime = (document.mime_type or "").strip()
        file_name = document.file_name or "document"
        media_kind = _document_media_kind(mime, file_name)
        if media_kind is not None:
            if (
                document.file_size is not None
                and document.file_size > MAX_MEDIA_SIZE_MB * 1024 * 1024
            ):
                await message.answer(
                    f"Файл слишком большой. Лимит Telegram Bot API — "
                    f"{MAX_MEDIA_SIZE_MB} МБ."
                )
                return
            try:
                data = await download_telegram_file(bot, document.file_id)
            except VisionError as exc:
                await message.answer(exc.message)
                return
            if not data:
                await message.answer("Пустой файл.")
                return
            logger.info(
                "document_media",
                kind=media_kind,
                name=file_name,
                mime=mime or "unknown",
                size=len(data),
            )
            await _process_attached_media(
                message,
                assistant,
                topic_storage,
                assistant_storage,
                state,
                data=data,
                filename=file_name,
                mime_type=mime
                or ("video/mp4" if media_kind == "video" else "audio/mpeg"),
                kind=media_kind,
            )
            return

    extracted = extract_image_from_message(message)
    if isinstance(extracted, str):
        await message.answer(extracted)
        return
    file_id, mime, file_name = extracted
    if (
        document is not None
        and document.file_size is not None
        and document.file_size > MAX_IMAGE_SIZE_MB * 1024 * 1024
    ):
        await message.answer(
            f"Файл слишком большой. Максимум — {MAX_IMAGE_SIZE_MB} МБ."
        )
        return
    user_media: list[UserMedia] | None = None
    try:
        data = await download_telegram_file(bot, file_id)
        ensure_within_size_limit(data, label="Файл")
        caption = message.caption or ""
        if mime == "application/pdf":
            pages = process_pdf(data)
            logger.info(
                "PDF, chat_id=%s name=%s size=%s pages=%s",
                message.chat.id,
                file_name,
                len(data),
                len(pages),
            )
            image_parts = [build_image_content(page, "image/png") for page in pages]
            multimodal = build_multimodal_content(caption, image_parts)
            placeholder = f"[Документ: {file_name}, {len(pages)} стр.]"
            status_text = "Анализирую документ…"
        else:
            logger.info(
                "Документ-картинка, chat_id=%s name=%s mime=%s size=%s",
                message.chat.id,
                file_name,
                mime,
                len(data),
            )
            image_parts = [build_image_content(data, mime)]
            multimodal = build_multimodal_content(caption, image_parts)
            placeholder = (
                f"[Изображение: {caption.strip() or file_name or 'без подписи'}]"
            )
            status_text = "Анализирую изображение…"
            user_media = [
                UserMedia(
                    kind="image",
                    filename=file_name or "image.bin",
                    mime_type=mime,
                    data=data,
                )
            ]
    except VisionError as exc:
        await message.answer(exc.message)
        return
    await _process_vision_message(
        message,
        assistant,
        topic_storage,
        assistant_storage,
        state,
        multimodal_content=multimodal,
        history_placeholder=placeholder,
        status_text=status_text,
        user_media=user_media,
    )


async def _rollback_user(
    chat_id: int,
    assistant_id: str,
    storage: TopicHistoryStorage,
    topic_id: str,
    message: Message,
) -> None:
    try:
        await storage.rollback_last_user(chat_id, assistant_id, topic_id)
    except TopicStorageError:
        await _reply_history_error(message)


async def _report_stream_error(
    message: Message,
    status: Message,
    parts: list[str],
    text: str,
    state: FSMContext,
) -> None:
    if parts:
        await _delete_bot_message(status)
        await message.answer(text)
        return
    edited = await _edit_stream(status, text, "")
    if edited:
        await _append_chrome(state, [status.message_id])
        return
    await _delete_bot_message(status)
    await message.answer(text)


def _model_keyboard(
    models: tuple[str, ...],
    prefix: str,
    *,
    include_default: bool = True,
) -> InlineKeyboardMarkup:
    """Кнопки моделей. В callback кладётся индекс, не имя."""
    rows = [
        [InlineKeyboardButton(text=model, callback_data=f"{prefix}:{index}")]
        for index, model in enumerate(models)
    ]
    if include_default:
        rows.append(
            [
                InlineKeyboardButton(
                    text="По умолчанию", callback_data=f"{prefix}:default"
                )
            ]
        )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _chosen_model_line(selected: str | None, fallback: str) -> str:
    if selected is None:
        return f"{fallback} (по умолчанию)"
    return selected


async def on_set_text_model(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Записывает текстовую модель в активный профиль."""
    await _apply_model_choice(
        query,
        assistant,
        assistant_storage,
        state,
        catalog=AVAILABLE_TEXT_MODELS,
        prefix="set_text:",
        kind="text",
    )


async def on_set_image_model(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Записывает модель изображений в активный профиль."""
    await _apply_model_choice(
        query,
        assistant,
        assistant_storage,
        state,
        catalog=AVAILABLE_IMAGE_MODELS,
        prefix="set_image:",
        kind="image",
    )


async def _apply_model_choice(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    state: FSMContext,
    *,
    catalog: tuple[str, ...],
    prefix: str,
    kind: str,
) -> None:
    message = query.message
    if not isinstance(message, Message):
        await query.answer()
        return
    raw = (query.data or "").removeprefix(prefix)
    if raw == "default":
        selected: str | None = None
    else:
        try:
            selected = catalog[int(raw)]
        except (ValueError, IndexError):
            await query.answer("Неизвестная модель", show_alert=True)
            return
    chat_id = message.chat.id
    try:
        profile = await _active_profile(chat_id, assistant_storage)
        if kind == "image":
            old_model = _image_model_of(profile, assistant)
            new_model = selected or assistant.config.default_image_model
            fields: dict[str, Any] = {"image_model": selected}
            if image_model_family(old_model) != image_model_family(new_model):
                fields["image_size"] = _default_image_size(new_model)
            updated = profile.model_copy(update=fields)
        else:
            updated = profile.model_copy(update={"text_model": selected})
        await assistant_storage.save_assistant(chat_id, updated)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    await query.answer("Модель обновлена")
    if kind == "image":
        text, markup = _model_picker_parts(updated, assistant, "image")
        await _edit_inline(message, text, markup)
        return
    await _return_to_settings(message, state, updated, assistant)


async def _return_to_settings(
    message: Message,
    state: FSMContext,
    profile: AssistantProfile,
    assistant: AssistAIClient,
) -> None:
    """Убирает выбор модели и обновляет предыдущий экран настроек."""
    text, markup = _settings_view(profile, assistant)
    try:
        await message.delete()
    except TelegramBadRequest as exc:
        logger.warning("Не удалось удалить выбор модели: %s", exc)
    data = await state.get_data()
    if _SCREEN_PICKER in data:
        data.pop(_SCREEN_PICKER, None)
        await state.set_data(data)
    settings_id = data.get(_SCREEN_SETTINGS)
    bot = message.bot
    if (
        isinstance(settings_id, int)
        and bot is not None
        and settings_id != message.message_id
    ):
        try:
            await bot.edit_message_text(
                text,
                chat_id=message.chat.id,
                message_id=settings_id,
                reply_markup=markup,
            )
            return
        except TelegramBadRequest as exc:
            lowered = str(exc).lower()
            if "message is not modified" in lowered:
                return
            logger.warning("Не удалось вернуть настройки: %s", exc)
    sent = await message.answer(text, reply_markup=markup)
    await state.update_data({_SCREEN_SETTINGS: sent.message_id})


def _clip_label(prefix: str, name: str) -> str:
    """Подпись кнопки не длиннее 64 символов."""
    label = f"{prefix}{name}"
    if len(label) <= 64:
        return label
    return label[:61] + "..."


def _optional_model(value: Any) -> str | None:
    """Пустая строка и не-строка означают модель по умолчанию."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _query_message(query: CallbackQuery) -> Message | None:
    """Сообщение кнопки. Служебные callback без сообщения пропускаются."""
    message = query.message
    if isinstance(message, Message):
        return message
    return None


def _release_outgoing(token: Token[bool] | None) -> Token[bool] | None:
    """Снимает запрет учёта исходящих сообщений, если он ещё включён."""
    if token is not None:
        _KEEP_OUTGOING.reset(token)
    return None


def _is_menu_button(text: str) -> bool:
    """Нижние кнопки и их старые подписи. Это не вопрос к персоне."""
    if text.startswith((f"{_MENU_ASSISTANTS} (", f"{_MENU_TOPICS} (")):
        return True
    return text in {
        _MENU_SETTINGS,
        MAIN_MENU_TEXT["assistants"],
        MAIN_MENU_TEXT["topics"],
        MAIN_MENU_TEXT["settings"],
        MAIN_MENU_TEXT["image"],
    }


async def _append_chrome(state: FSMContext, ids: list[int]) -> None:
    """Запоминает служебные сообщения бота, чтобы позже убрать их из чата."""
    if not ids:
        return
    data = await state.get_data()
    current = data.get(_CHROME)
    merged: list[int] = []
    if isinstance(current, list):
        merged.extend(item for item in current if isinstance(item, int))
    for item in ids:
        if item not in merged:
            merged.append(item)
    if len(merged) > 50:
        merged = merged[-50:]
    await state.update_data({_CHROME: merged})


async def _reset_flow(state: FSMContext) -> None:
    """Сбрасывает шаг мастера и сохраняет id сообщения, которое держит меню."""
    data = await state.get_data()
    anchor = data.get(_MENU_ANCHOR)
    chrome = data.get(_CHROME)
    await state.clear()
    kept: dict[str, Any] = {}
    if isinstance(anchor, int):
        kept[_MENU_ANCHOR] = anchor
    if isinstance(chrome, list):
        kept[_CHROME] = chrome
    if kept:
        await state.update_data(kept)


async def _delete_bot_message(message: Message) -> None:
    """Удаляет сообщение бота. Повторное удаление не считается сбоем."""
    try:
        await message.delete()
    except TelegramBadRequest as exc:
        logger.warning("Не удалось удалить сообщение: %s", exc)


async def _apply_reply_menu(
    message: Message, menu: ReplyKeyboardMarkup, state: FSMContext
) -> None:
    """Ставит нижние кнопки.

    Сообщение с клавиатурой нельзя удалять: вместе с ним клиент убирает меню.
    Текст невидимый, чтобы в чате не было отдельной реплики. Прошлое такое
    сообщение удаляется, когда новое уже держит меню.
    """
    token = _KEEP_OUTGOING.set(True)
    try:
        sent = await message.answer("\u2060", reply_markup=menu)
    finally:
        _KEEP_OUTGOING.reset(token)
    data = await state.get_data()
    previous = data.get(_MENU_ANCHOR)
    await state.update_data({_MENU_ANCHOR: sent.message_id})
    if (
        isinstance(previous, int)
        and previous != sent.message_id
        and message.bot is not None
    ):
        try:
            await message.bot.delete_message(
                chat_id=message.chat.id, message_id=previous
            )
        except TelegramBadRequest as exc:
            logger.warning("Не удалось убрать прежнее сообщение меню: %s", exc)


async def _sweep_chrome(
    message: Message, state: FSMContext, data: dict[str, Any]
) -> None:
    """Удаляет меню, настройки и прочие служебные реплики.

    Перед удалением заново ставит нижние кнопки: иначе клиент прячет меню
    вместе с последним сообщением, на котором оно висело.
    """
    stored = await state.get_data()
    raw_ids = stored.get(_CHROME)
    if not isinstance(raw_ids, list) or not raw_ids:
        return
    profiles = data.get("assistant_storage")
    topics = data.get("topic_storage")
    if isinstance(profiles, AssistantStorage) and isinstance(
        topics, TopicHistoryStorage
    ):
        menu = await _menu_for(message.chat.id, profiles, topics)
        await _apply_reply_menu(message, menu, state)
    stored = await state.get_data()
    anchor = stored.get(_MENU_ANCHOR)
    bot = message.bot
    if bot is not None:
        for item in raw_ids:
            if not isinstance(item, int) or item == anchor:
                continue
            try:
                await bot.delete_message(chat_id=message.chat.id, message_id=item)
            except TelegramBadRequest:
                logger.debug(
                    "Служебное сообщение уже удалено chat_id=%s message_id=%s",
                    message.chat.id,
                    item,
                )
    refreshed = await state.get_data()
    for key in _SCREEN_KEYS:
        refreshed.pop(key, None)
    refreshed[_CHROME] = []
    await state.set_data(refreshed)


async def _cleanup_wizard_ui(
    message: Message,
    state: FSMContext,
    assistant_storage: AssistantStorage,
    topic_storage: TopicHistoryStorage,
    *,
    also_delete: list[int] | None = None,
) -> None:
    """Сразу убирает UI мастера. Нужно после шагов, которые заканчиваются callback.

    Обычная очистка срабатывает только на Message; «Готово» при создании
    персоны — callback, поэтому экран моделей иначе остаётся в чате.
    """
    stored = await state.get_data()
    merged: list[int] = []
    raw_ids = stored.get(_CHROME)
    if isinstance(raw_ids, list):
        merged.extend(item for item in raw_ids if isinstance(item, int))
    for item in also_delete or []:
        if item not in merged:
            merged.append(item)
    await state.update_data({_CHROME: merged})
    await _sweep_chrome(
        message,
        state,
        {
            "assistant_storage": assistant_storage,
            "topic_storage": topic_storage,
        },
    )


async def _drop_user_noise(message: Message, state: FSMContext) -> None:
    """Убирает команды, нажатия меню и шаги мастера. Вопрос к персоне остаётся."""
    text = message.text or ""
    if not text:
        return
    current = await state.get_state()
    if current is None and not text.startswith("/") and not _is_menu_button(text):
        return
    try:
        await message.delete()
    except TelegramBadRequest as exc:
        logger.warning("Не удалось убрать служебную реплику пользователя: %s", exc)


class _CleanChat(BaseMiddleware):
    """Перед новым сообщением убирает из чата всё, кроме диалога и картинок."""

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        state = data.get("state")
        bucket: list[int] = []
        token = _CHROME_BUCKET.set(bucket)
        try:
            if isinstance(event, Message) and isinstance(state, FSMContext):
                current = await state.get_state()
                if current is not None and (event.photo or event.document):
                    keep = _KEEP_OUTGOING.set(True)
                    try:
                        await event.answer(
                            "Сначала завершите текущий шаг или нажмите /cancel."
                        )
                    finally:
                        _release_outgoing(keep)
                    return None
                await _sweep_chrome(event, state, data)
                await _drop_user_noise(event, state)
            return await handler(event, data)
        finally:
            if isinstance(state, FSMContext):
                await _append_chrome(state, bucket)
            _CHROME_BUCKET.reset(token)


async def _close_screen(message: Message, state: FSMContext, key: str) -> None:
    """Удаляет экран с кнопками и забывает его id, чтобы не править пустое место."""
    try:
        await message.delete()
    except TelegramBadRequest as exc:
        logger.warning("Не удалось удалить экран: %s", exc)
    data = await state.get_data()
    if data.get(key) == message.message_id:
        data.pop(key, None)
        await state.set_data(data)


async def _edit_inline(
    message: Message, text: str, markup: InlineKeyboardMarkup | None
) -> None:
    """Правит текст и кнопки. Повтор того же текста Telegram не считает ошибкой."""
    try:
        await message.edit_text(text, reply_markup=markup)
    except TelegramBadRequest as exc:
        if "message is not modified" not in str(exc).lower():
            logger.warning("Не удалось обновить сообщение: %s", exc)


async def _drop_flow(state: FSMContext) -> None:
    """Выходит из пошагового ввода и сохраняет id уже открытых экранов."""
    data = await state.get_data()
    kept = {key: value for key, value in data.items() if key not in _FLOW_KEYS}
    await state.set_state(None)
    await state.set_data(kept)


async def _upsert_screen(
    message: Message,
    state: FSMContext,
    key: str,
    text: str,
    markup: InlineKeyboardMarkup,
) -> None:
    """Повторный вызов экрана правит прежнее сообщение, иначе отправляет новое."""
    data = await state.get_data()
    message_id = data.get(key)
    bot = message.bot
    if isinstance(message_id, int) and bot is not None:
        try:
            await bot.edit_message_text(
                text,
                chat_id=message.chat.id,
                message_id=message_id,
                reply_markup=markup,
            )
            return
        except TelegramBadRequest as exc:
            lowered = str(exc).lower()
            if "message is not modified" in lowered:
                return
            logger.warning("Не удалось обновить экран: %s", exc)
    sent = await message.answer(text, reply_markup=markup)
    await state.update_data({key: sent.message_id})


def _page_bounds(count: int, page: int) -> tuple[int, int]:
    """Номер страницы в допустимых пределах и общее число страниц."""
    total = max(1, (count + PAGE_SIZE - 1) // PAGE_SIZE)
    current = max(0, min(page, total - 1))
    return current, total


def _pager_row(page: int, total: int, prefix: str) -> list[InlineKeyboardButton] | None:
    """Строка ◀ / ▶. На одной странице её нет."""
    if total <= 1:
        return None
    row: list[InlineKeyboardButton] = []
    if page > 0:
        row.append(InlineKeyboardButton(text="◀", callback_data=f"{prefix}:{page - 1}"))
    if page < total - 1:
        row.append(InlineKeyboardButton(text="▶", callback_data=f"{prefix}:{page + 1}"))
    return row or None


def _assistants_text(
    profiles: list[AssistantProfile], active_id: str, page: int = 0
) -> str:
    """Текст экрана персон: кто активен и номер страницы. Имена только на кнопках."""
    current, total = _page_bounds(len(profiles), page)
    active = next((item for item in profiles if item.id == active_id), None)
    active_name = active.name if active is not None else "—"
    lines = [f"Активный: ✅ {active_name}"]
    if not profiles:
        lines.append("")
        lines.append("Персон пока нет.")
    lines.append("")
    lines.append(f"Стр. {current + 1}/{total}")
    return "\n".join(lines)


def _assistants_keyboard(
    profiles: list[AssistantProfile], active_id: str, page: int = 0
) -> InlineKeyboardMarkup:
    """Кнопки персон текущей страницы и переход по страницам."""
    current, total = _page_bounds(len(profiles), page)
    chunk = profiles[current * PAGE_SIZE : (current + 1) * PAGE_SIZE]
    rows: list[list[InlineKeyboardButton]] = []
    for profile in chunk:
        mark = "✅ " if profile.id == active_id else ""
        rows.append(
            [
                InlineKeyboardButton(
                    text=_clip_label(mark, profile.name),
                    callback_data=f"asst:use:{profile.id}",
                )
            ]
        )
    pager = _pager_row(current, total, "asst_page")
    if pager is not None:
        rows.append(pager)
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _assistants_markup(
    profiles: list[AssistantProfile], active_id: str, page: int = 0
) -> tuple[str, InlineKeyboardMarkup]:
    """Текст и кнопки экрана персон. Здесь только переключение."""
    return (
        _assistants_text(profiles, active_id, page),
        _assistants_keyboard(profiles, active_id, page),
    )


async def _assistant_screen(
    chat_id: int, storage: AssistantStorage
) -> tuple[str, InlineKeyboardMarkup]:
    """Собирает экран персон из хранилища."""
    profiles = await storage.list_assistants(chat_id)
    active = await storage.get_active_assistant(chat_id)
    active_id = active.id if active is not None else ""
    return _assistants_markup(profiles, active_id)


async def _open_assistants(
    message: Message, storage: AssistantStorage, state: FSMContext
) -> None:
    """Открывает или обновляет экран персон."""
    try:
        text, markup = await _assistant_screen(message.chat.id, storage)
    except AssistantStorageError:
        await _reply_profile_error(message)
        return
    await _upsert_screen(message, state, _SCREEN_ASSISTANTS, text, markup)


def _topics_text(topics: list[dict[str, str]], active: str, page: int = 0) -> str:
    """Текст экрана тем: какая активна и номер страницы. Имена только на кнопках."""
    current, total = _page_bounds(len(topics), page)
    active_name = next((item["name"] for item in topics if item["id"] == active), "нет")
    lines = [f"Активная: 📍 {active_name}"]
    if not topics:
        lines.append("")
        lines.append("Тем пока нет.")
    lines.append("")
    lines.append(f"Стр. {current + 1}/{total}")
    return "\n".join(lines)


def _topics_keyboard(
    topics: list[dict[str, str]], active: str, page: int = 0
) -> InlineKeyboardMarkup:
    """Кнопки тем текущей страницы: переключить, переименовать, удалить."""
    current, total = _page_bounds(len(topics), page)
    chunk = topics[current * PAGE_SIZE : (current + 1) * PAGE_SIZE]
    rows: list[list[InlineKeyboardButton]] = []
    for topic in chunk:
        name = topic["name"]
        topic_id = topic["id"]
        mark = "✅ " if topic_id == active else ""
        rows.append(
            [
                InlineKeyboardButton(
                    text=_clip_label(mark, name),
                    callback_data=f"topic:use:{topic_id}",
                ),
                InlineKeyboardButton(
                    text="✏️", callback_data=f"topic:rename:{topic_id}"
                ),
                InlineKeyboardButton(
                    text="🗑", callback_data=f"topic_del_ask:{topic_id}"
                ),
            ]
        )
    rows.append([InlineKeyboardButton(text="Новая тема", callback_data="topic:new")])
    pager = _pager_row(current, total, "topic_page")
    if pager is not None:
        rows.append(pager)
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _topics_markup(
    topics: list[dict[str, str]], active_id: str, page: int = 0
) -> tuple[str, InlineKeyboardMarkup]:
    """Текст и кнопки экрана тем."""
    return (
        _topics_text(topics, active_id, page),
        _topics_keyboard(topics, active_id, page),
    )


async def _topics_screen(
    chat_id: int,
    assistant_id: str,
    storage: TopicHistoryStorage,
) -> tuple[str, InlineKeyboardMarkup]:
    """Собирает экран тем активной персоны."""
    topics = await storage.list_topics(chat_id, assistant_id)
    active = await storage.get_active_topic(chat_id, assistant_id)
    return _topics_markup(topics, active)


async def _open_topics(
    message: Message,
    storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Открывает или обновляет экран тем активной персоны."""
    try:
        profile = await _active_profile(message.chat.id, assistant_storage)
        text, markup = await _topics_screen(message.chat.id, profile.id, storage)
    except AssistantStorageError:
        await _reply_profile_error(message)
        return
    except TopicStorageError:
        await _reply_history_error(message)
        return
    await _upsert_screen(message, state, _SCREEN_TOPICS, text, markup)


def _model_picker_parts(
    profile: AssistantProfile, assistant: AssistAIClient, kind: str
) -> tuple[str, InlineKeyboardMarkup]:
    """Текст и кнопки выбора модели активной персоны."""
    if kind == "image":
        current = _chosen_model_line(
            profile.image_model, assistant.config.default_image_model
        )
        shown_size = _display_image_size(profile, assistant)
        image_model = _image_model_of(profile, assistant)
        rows: list[list[InlineKeyboardButton]] = []
        for index, model in enumerate(AVAILABLE_IMAGE_MODELS):
            mark = "✅ " if image_model == model else ""
            rows.append(
                [
                    InlineKeyboardButton(
                        text=_clip_label(mark, model),
                        callback_data=f"set_image:{index}",
                    )
                ]
            )
        size_row: list[InlineKeyboardButton] = []
        for item in _size_choices(image_model):
            mark = "✅ " if item == shown_size else ""
            size_row.append(
                InlineKeyboardButton(
                    text=f"{mark}{item}", callback_data=f"set_size:{item}"
                )
            )
        rows.append(size_row)
        rows.append([InlineKeyboardButton(text="Готово", callback_data="picker:done")])
        text = (
            f"Модель картинок ассистента «{profile.name}»:\n"
            f"{current}\n"
            f"Размер: {shown_size}\n\n"
            "Выберите модель или размер:"
        )
        return text, InlineKeyboardMarkup(inline_keyboard=rows)
    current = _chosen_model_line(
        profile.text_model, assistant.config.default_text_model
    )
    keyboard = _model_keyboard(AVAILABLE_TEXT_MODELS, "set_text")
    text = (
        f"Текстовая модель ассистента «{profile.name}»:\n{current}\n\nВыберите модель:"
    )
    return text, keyboard


def _voice_settings_lines(
    profile: AssistantProfile, assistant: AssistAIClient
) -> list[str]:
    """Текст блока голосовых настроек персоны."""
    tts_model = profile.tts_model or assistant.config.default_tts_model
    tts_voice = profile.tts_voice or assistant.config.default_tts_voice
    tts_format = profile.tts_format or assistant.config.default_tts_format
    stt_language = profile.stt_language or "авто"
    mode_label = _VOICE_MODE_LABELS.get(profile.voice_mode, profile.voice_mode)
    return [
        "🎙 Голос:",
        f"Режим: {mode_label}",
        f"TTS модель: {tts_model}",
        f"Голос: {tts_voice}",
        f"Формат: {tts_format}",
        f"STT язык: {stt_language}",
    ]


def _voice_view(
    profile: AssistantProfile, assistant: AssistAIClient
) -> tuple[str, InlineKeyboardMarkup]:
    """Экран голосового ответа: режим, голос TTS, язык STT."""
    lines = [
        f"Голосовой ответ — «{profile.name}»",
        "",
        *_voice_settings_lines(profile, assistant),
    ]
    text = "\n".join(lines)
    rows: list[list[InlineKeyboardButton]] = []
    mode_row: list[InlineKeyboardButton] = []
    for mode in ("off", "auto", "always"):
        mark = "✅ " if profile.voice_mode == mode else ""
        mode_row.append(
            InlineKeyboardButton(
                text=f"{mark}{_VOICE_MODE_LABELS[mode]}",
                callback_data=f"voice_mode:{mode}",
            )
        )
    rows.append(mode_row)
    voice_row: list[InlineKeyboardButton] = []
    current_voice = profile.tts_voice or assistant.config.default_tts_voice
    for index, name in enumerate(TTS_VOICES):
        mark = "✅ " if current_voice == name else ""
        voice_row.append(
            InlineKeyboardButton(
                text=f"{mark}{name}",
                callback_data=f"voice_name:{index}",
            )
        )
    rows.append(voice_row)
    lang_row: list[InlineKeyboardButton] = []
    for label, value in STT_LANGUAGES:
        mark = "✅ " if profile.stt_language == value else ""
        lang_row.append(
            InlineKeyboardButton(
                text=f"{mark}STT {label}",
                callback_data=f"stt_lang:{label}",
            )
        )
    rows.append(lang_row)
    rows.append([InlineKeyboardButton(text="Готово", callback_data="settings:open")])
    return text, InlineKeyboardMarkup(inline_keyboard=rows)


def _settings_view(
    profile: AssistantProfile, assistant: AssistAIClient
) -> tuple[str, InlineKeyboardMarkup]:
    """Настройки активной персоны: имя, промпт, модели, голос."""
    shown_size = _display_image_size(profile, assistant)
    mode_label = _VOICE_MODE_LABELS.get(profile.voice_mode, profile.voice_mode)
    lines = [
        f"Текущая персона: ✅ {profile.name}",
        (
            "Текстовая модель: "
            f"{_chosen_model_line(profile.text_model, assistant.config.default_text_model)}"
        ),
        (
            "Модель картинок: "
            f"{_chosen_model_line(profile.image_model, assistant.config.default_image_model)}"
        ),
        f"Размер картинки: {shown_size}",
        f"Голосовой ответ: {mode_label}",
    ]
    text = "\n".join(lines)
    rows: list[list[InlineKeyboardButton]] = [
        [
            InlineKeyboardButton(
                text="Изменить имя",
                callback_data=f"asst_rename:{profile.id}",
            )
        ],
        [
            InlineKeyboardButton(
                text="Изменить промпт",
                callback_data=f"asst_edit:{profile.id}",
            )
        ],
        [
            InlineKeyboardButton(
                text="Выбрать текстовую модель",
                callback_data="set_text_menu",
            )
        ],
        [
            InlineKeyboardButton(
                text="Выбрать модель картинок",
                callback_data="set_image_menu",
            )
        ],
        [
            InlineKeyboardButton(
                text="Голосовой ответ",
                callback_data="voice_menu",
            )
        ],
    ]
    if profile.id != DEFAULT_ASSISTANT_ID:
        rows.append(
            [
                InlineKeyboardButton(
                    text="Удалить ассистента",
                    callback_data=f"asst:del_ask:{profile.id}",
                )
            ]
        )
    rows.append(
        [
            InlineKeyboardButton(
                text="Создать нового ассистента",
                callback_data="asst:new",
            )
        ]
    )
    rows.append([InlineKeyboardButton(text="Готово", callback_data="settings:close")])
    return text, InlineKeyboardMarkup(inline_keyboard=rows)


def _image_view(
    selected: str | None, size: str, assistant: AssistAIClient
) -> tuple[str, InlineKeyboardMarkup]:
    """Экран генерации: модель, размер семьи модели и переход к описанию."""
    label = selected or assistant.config.default_image_model
    allowed = _size_choices(label)
    shown = size if size in allowed else _default_image_size(label)
    text = f"Генерация изображения\nМодель: {label}\nРазмер: {shown}"
    rows: list[list[InlineKeyboardButton]] = []
    for index, model in enumerate(AVAILABLE_IMAGE_MODELS):
        mark = "✅ " if selected == model else ""
        rows.append(
            [
                InlineKeyboardButton(
                    text=_clip_label(mark, model),
                    callback_data=f"image:model:{index}",
                )
            ]
        )
    default_mark = "✅ " if selected is None else ""
    rows.append(
        [
            InlineKeyboardButton(
                text=f"{default_mark}По умолчанию",
                callback_data="image:model:default",
            )
        ]
    )
    size_row: list[InlineKeyboardButton] = []
    for item in allowed:
        mark = "✅ " if item == shown else ""
        size_row.append(
            InlineKeyboardButton(
                text=f"{mark}{item}", callback_data=f"image:size:{item}"
            )
        )
    rows.append(size_row)
    rows.append(
        [InlineKeyboardButton(text="✍️ Описать картинку", callback_data="image:go")]
    )
    return text, InlineKeyboardMarkup(inline_keyboard=rows)


def _draft_models_text(data: dict[str, Any], assistant: AssistAIClient) -> str:
    """Текущий выбор моделей на шаге создания персоны."""
    text_model = _optional_model(data.get("flow_text_model"))
    image_model = _optional_model(data.get("flow_image_model"))
    text_line = (
        f"{assistant.config.default_text_model} (по умолчанию)"
        if text_model is None
        else text_model
    )
    image_line = (
        f"{assistant.config.default_image_model} (по умолчанию)"
        if image_model is None
        else image_model
    )
    return (
        "Выберите модели (необязательно) или нажмите «Готово».\n"
        f"Текст: {text_line}\n"
        f"Картинка: {image_line}"
    )


def _draft_models_keyboard(data: dict[str, Any]) -> InlineKeyboardMarkup:
    """Кнопки временного выбора моделей и подтверждение создания."""
    text_model = _optional_model(data.get("flow_text_model"))
    image_model = _optional_model(data.get("flow_image_model"))
    rows: list[list[InlineKeyboardButton]] = []
    for index, model in enumerate(AVAILABLE_TEXT_MODELS):
        mark = "✅ " if text_model == model else ""
        rows.append(
            [
                InlineKeyboardButton(
                    text=_clip_label(mark, model),
                    callback_data=f"asst:new_text:{index}",
                )
            ]
        )
    for index, model in enumerate(AVAILABLE_IMAGE_MODELS):
        mark = "✅ " if image_model == model else ""
        rows.append(
            [
                InlineKeyboardButton(
                    text=_clip_label(f"{mark}🖼 ", model),
                    callback_data=f"asst:new_image:{index}",
                )
            ]
        )
    rows.append(
        [InlineKeyboardButton(text="По умолчанию", callback_data="asst:new_default")]
    )
    rows.append(
        [InlineKeyboardButton(text="✅ Готово", callback_data="asst:new_commit")]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _confirm_row(
    yes_text: str, yes_data: str, cancel_data: str
) -> InlineKeyboardMarkup:
    """Пара кнопок: подтвердить действие или вернуться к списку."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text=yes_text, callback_data=yes_data),
                InlineKeyboardButton(text="← Отмена", callback_data=cancel_data),
            ]
        ]
    )


async def _generate_and_send(
    message: Message,
    assistant: AssistAIClient,
    prompt: str,
    model: str | None,
    size: str = "1024x1024",
    *,
    reply_markup: ReplyKeyboardMarkup | None = None,
) -> None:
    """Просит ProxyAPI нарисовать картинку и отправляет её в чат."""
    user_id = message.from_user.id if message.from_user else message.chat.id
    bind_request_context(user_id=user_id, chat_id=message.chat.id)
    note_active_user(user_id)
    logger.info(
        "image_generation",
        model=model or assistant.config.default_image_model,
        size=size,
        prompt_chars=len(prompt),
    )
    set_billing_user(user_id)
    keep = _KEEP_OUTGOING.set(True)
    try:
        progress = await message.answer("Генерирую изображение…")
    finally:
        _KEEP_OUTGOING.reset(keep)
    try:
        await cost_tracker.check_budget(user_id, cost_tracker.estimate_image_cost(1))
        images = await job_queue.generate_image(assistant, prompt, model, size)
        await _delete_bot_message(progress)
        if not images:
            await message.answer(MSG_IMAGE_FAILED, reply_markup=reply_markup)
            return
        for index, image in enumerate(images):
            await _send_generated_image(message, image, index)
    except BudgetExceededError as exc:
        await _delete_bot_message(progress)
        await message.answer(exc.message, reply_markup=reply_markup)
    except AssistAIError as exc:
        logger.error("AssistAI ошибка: %s", exc.message)
        await _delete_bot_message(progress)
        await message.answer(MSG_IMAGE_FAILED, reply_markup=reply_markup)
    except httpx.TimeoutException:
        logger.error("Таймаут генерации изображения")
        await _delete_bot_message(progress)
        await message.answer(MSG_TIMEOUT, reply_markup=reply_markup)
    except Exception:
        logger.exception("Непредвиденная ошибка генерации изображения")
        await _delete_bot_message(progress)
        await message.answer(MSG_INTERNAL, reply_markup=reply_markup)


async def on_menu_assistants(
    message: Message, assistant_storage: AssistantStorage, state: FSMContext
) -> None:
    """Кнопка «Ассистенты» открывает список персон."""
    await _drop_flow(state)
    await _open_assistants(message, assistant_storage, state)


async def on_menu_topics(
    message: Message,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Кнопка «Темы» открывает темы активной персоны."""
    await _drop_flow(state)
    await _open_topics(message, topic_storage, assistant_storage, state)


async def _open_settings(
    message: Message,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Открывает или обновляет экран настроек активной персоны."""
    try:
        profile = await _active_profile(message.chat.id, assistant_storage)
    except AssistantStorageError:
        await _reply_profile_error(message)
        return
    text, markup = _settings_view(profile, assistant)
    await _upsert_screen(message, state, _SCREEN_SETTINGS, text, markup)


async def on_menu_settings(
    message: Message,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Кнопка «Настройки» показывает модели активной персоны."""
    await _drop_flow(state)
    await _open_settings(message, assistant, assistant_storage, state)


async def on_menu_image(
    message: Message,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Кнопка «Картинка» открывает выбор модели и размера."""
    await _drop_flow(state)
    try:
        profile = await _active_profile(message.chat.id, assistant_storage)
    except AssistantStorageError:
        await _reply_profile_error(message)
        return
    selected = profile.image_model
    model_id = selected or assistant.config.default_image_model
    size = _default_image_size(model_id)
    await state.set_state(ImageSG.config)
    await state.update_data(image_model=selected, image_size=size)
    text, markup = _image_view(selected, size, assistant)
    await _upsert_screen(message, state, _SCREEN_IMAGE, text, markup)


async def on_asst_list(
    query: CallbackQuery, assistant_storage: AssistantStorage, state: FSMContext
) -> None:
    """Возвращает сообщение к списку персон."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    try:
        text, markup = await _assistant_screen(message.chat.id, assistant_storage)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    await _edit_inline(message, text, markup)
    await state.update_data({_SCREEN_ASSISTANTS: message.message_id})
    await query.answer()


async def on_asst_use(
    query: CallbackQuery,
    assistant_storage: AssistantStorage,
    topic_storage: TopicHistoryStorage,
    state: FSMContext,
) -> None:
    """Делает выбранную персону активной, убирает экран и обновляет нижнее меню."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    profile_id = (query.data or "").removeprefix("asst:use:")
    chat_id = message.chat.id
    try:
        await assistant_storage.set_active_assistant(chat_id, profile_id)
        menu = await _menu_for(chat_id, assistant_storage, topic_storage)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    except TopicStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    await query.answer()
    await _close_screen(message, state, _SCREEN_ASSISTANTS)
    await _apply_reply_menu(message, menu, state)
    logger.info("Активный ассистент chat_id=%s id=%s", chat_id, profile_id)


async def on_assistants_page(
    query: CallbackQuery, assistant_storage: AssistantStorage
) -> None:
    """Открывает другую страницу списка персон."""
    message = query.message
    if not isinstance(message, Message):
        await query.answer()
        return
    try:
        page = int((query.data or "").removeprefix("asst_page:"))
    except ValueError:
        await query.answer("Неверная страница", show_alert=True)
        return
    try:
        profiles = await assistant_storage.list_assistants(message.chat.id)
        active = await assistant_storage.get_active_assistant(message.chat.id)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    active_id = active.id if active is not None else ""
    await query.answer()
    await _edit_or_answer(
        query,
        _assistants_text(profiles, active_id, page),
        _assistants_keyboard(profiles, active_id, page),
    )


async def on_asst_rename_ask(
    query: CallbackQuery, state: FSMContext, assistant_storage: AssistantStorage
) -> None:
    """Просит новое имя активной или выбранной персоны."""
    message = query.message
    if not isinstance(message, Message):
        await query.answer()
        return
    profile_id = (query.data or "").removeprefix("asst_rename:")
    try:
        profiles = await assistant_storage.list_assistants(message.chat.id)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    profile = next((item for item in profiles if item.id == profile_id), None)
    if profile is None:
        await query.answer("Ассистент не найден", show_alert=True)
        return
    await state.set_state(EditAssistantSG.name)
    await state.update_data(id=profile.id, name=profile.name)
    await query.answer()
    await _edit_or_answer(
        query,
        f"Текущее имя: «{profile.name}»\n\n"
        "Введите новое имя (1–50 символов). /cancel — отмена.",
    )


async def sg_edit_name(
    message: Message,
    state: FSMContext,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
) -> None:
    """Сохраняет новое имя выбранной персоны."""
    name = (message.text or "").strip()
    if not name or len(name) > 50:
        await message.answer("Имя: 1–50 символов. /cancel — отмена.")
        return
    data = await state.get_data()
    profile_id = data.get("id")
    if not isinstance(profile_id, str) or not profile_id:
        await _reset_flow(state)
        await message.answer("Ассистент не найден.")
        return
    try:
        profiles = await assistant_storage.list_assistants(message.chat.id)
        old = next((item for item in profiles if item.id == profile_id), None)
        if old is None:
            await _reset_flow(state)
            await message.answer("Ассистент не найден.")
            return
        updated = old.model_copy(update={"name": name})
        await assistant_storage.save_assistant(message.chat.id, updated)
    except AssistantStorageError:
        await _reply_profile_error(message)
        return
    except ValidationError:
        await message.answer("Имя: 1–50 символов. /cancel — отмена.")
        return
    await _reset_flow(state)
    logger.info(
        "Обновлено имя ассистента chat_id=%s id=%s", message.chat.id, updated.id
    )
    await message.answer(f"✅ Имя ассистента изменено на «{updated.name}».")
    await _open_settings(message, assistant, assistant_storage, state)


async def on_asst_edit_ask(
    query: CallbackQuery, state: FSMContext, assistant_storage: AssistantStorage
) -> None:
    """Показывает текущий промпт и ждёт новый текст."""
    message = query.message
    if not isinstance(message, Message):
        await query.answer()
        return
    profile_id = (query.data or "").removeprefix("asst_edit:")
    try:
        profiles = await assistant_storage.list_assistants(message.chat.id)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    profile = next((item for item in profiles if item.id == profile_id), None)
    if profile is None:
        await query.answer("Ассистент не найден", show_alert=True)
        return
    await state.set_state(EditAssistantSG.prompt)
    await state.update_data(id=profile.id, name=profile.name)
    await query.answer()
    await _edit_or_answer(
        query,
        f"Промпт ассистента «{profile.name}»:\n\n{profile.system_prompt}\n\n"
        "Пришлите новый промпт (≥10 символов). /cancel — отмена.",
    )


async def sg_edit_prompt(
    message: Message,
    state: FSMContext,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
) -> None:
    """Сохраняет новый системный промпт выбранной персоны."""
    prompt = message.text or ""
    if len(prompt) < 10:
        await message.answer("Промпт минимум 10 символов. /cancel — отмена.")
        return
    data = await state.get_data()
    profile_id = data.get("id")
    if not isinstance(profile_id, str) or not profile_id:
        await _reset_flow(state)
        await message.answer("Ассистент не найден.")
        return
    try:
        profiles = await assistant_storage.list_assistants(message.chat.id)
        old = next((item for item in profiles if item.id == profile_id), None)
        if old is None:
            await _reset_flow(state)
            await message.answer("Ассистент не найден.")
            return
        updated = old.model_copy(update={"system_prompt": prompt})
        await assistant_storage.save_assistant(message.chat.id, updated)
    except AssistantStorageError:
        await _reply_profile_error(message)
        return
    except ValidationError:
        await message.answer("Промпт минимум 10 символов. /cancel — отмена.")
        return
    await _reset_flow(state)
    logger.info(
        "Обновлён промпт ассистента chat_id=%s id=%s", message.chat.id, updated.id
    )
    await message.answer(f"✅ Промпт ассистента «{updated.name}» обновлён.")
    await _open_settings(message, assistant, assistant_storage, state)


async def on_asst_edit(
    query: CallbackQuery, assistant_storage: AssistantStorage, state: FSMContext
) -> None:
    """Запускает правку промпта существующей персоны."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    profile_id = (query.data or "").removeprefix("asst:edit:")
    try:
        profiles = await assistant_storage.list_assistants(message.chat.id)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    profile = next((item for item in profiles if item.id == profile_id), None)
    if profile is None:
        await query.answer("Ассистент не найден.", show_alert=True)
        return
    await _drop_flow(state)
    await state.set_state(NewAssistantSG.prompt)
    await state.update_data(
        flow_edit_id=profile.id,
        flow_name=profile.name,
        flow_text_model=profile.text_model,
        flow_image_model=profile.image_model,
    )
    await query.answer()
    await message.answer(
        f"Пришлите системный промпт для «{profile.name}» (мин. 10 символов).\n"
        "/cancel — отмена"
    )


async def on_asst_del_ask(
    query: CallbackQuery, assistant_storage: AssistantStorage
) -> None:
    """Спрашивает подтверждение перед удалением персоны."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    profile_id = (query.data or "").removeprefix("asst:del_ask:")
    if profile_id == DEFAULT_ASSISTANT_ID:
        await query.answer("Базового ассистента удалить нельзя.", show_alert=True)
        return
    try:
        profiles = await assistant_storage.list_assistants(message.chat.id)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    profile = next((item for item in profiles if item.id == profile_id), None)
    if profile is None:
        await query.answer("Ассистент не найден.", show_alert=True)
        return
    await query.answer()
    await _edit_inline(
        message,
        f"Удалить ассистента «{profile.name}»?",
        _confirm_row("✅ Да, удалить", f"asst:del_yes:{profile.id}", "settings:open"),
    )


async def on_asst_del_yes(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    topic_storage: TopicHistoryStorage,
    state: FSMContext,
) -> None:
    """Удаляет персону и её темы, затем показывает настройки оставшейся."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    profile_id = (query.data or "").removeprefix("asst:del_yes:")
    chat_id = message.chat.id
    try:
        profiles = await assistant_storage.list_assistants(chat_id)
        name = next(
            (item.name for item in profiles if item.id == profile_id), profile_id
        )
        if profile_id != DEFAULT_ASSISTANT_ID:
            await topic_storage.delete_assistant_topics(chat_id, profile_id)
        await assistant_storage.delete_assistant(chat_id, profile_id)
        profile = await _active_profile(chat_id, assistant_storage)
        menu = await _menu_for(chat_id, assistant_storage, topic_storage)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    except TopicStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    logger.info("Удалён ассистент chat_id=%s id=%s", chat_id, profile_id)
    await query.answer("Удалено")
    text, markup = _settings_view(profile, assistant)
    await _edit_inline(message, text, markup)
    await state.update_data({_SCREEN_SETTINGS: message.message_id})
    await message.answer(f"Ассистент «{name}» удалён.", reply_markup=menu)


async def on_asst_new_start(query: CallbackQuery, state: FSMContext) -> None:
    """Первый шаг создания персоны: запрос имени."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    await _drop_flow(state)
    await state.set_state(NewAssistantSG.name)
    await query.answer()
    await message.answer("Введите имя персоны (1–50 символов).\n/cancel — отмена")


async def sg_new_assistant_name(message: Message, state: FSMContext) -> None:
    """Принимает имя персоны и переходит к промпту."""
    name = (message.text or "").strip()
    if not name or len(name) > 50:
        await message.answer("Введите имя персоны (1–50 символов).\n/cancel — отмена")
        return
    await state.update_data(flow_name=name, flow_edit_id=None)
    await state.set_state(NewAssistantSG.prompt)
    await message.answer(
        "Пришлите системный промпт (мин. 10 символов).\n/cancel — отмена"
    )


async def sg_new_assistant_prompt(
    message: Message, state: FSMContext, assistant: AssistAIClient
) -> None:
    """Принимает промпт и показывает выбор моделей."""
    prompt = (message.text or "").strip()
    if len(prompt) < 10:
        await message.answer(
            "Пришлите системный промпт (мин. 10 символов).\n/cancel — отмена"
        )
        return
    await state.update_data(flow_prompt=prompt)
    await state.set_state(NewAssistantSG.models)
    data = await state.get_data()
    await message.answer(
        _draft_models_text(data, assistant),
        reply_markup=_draft_models_keyboard(data),
    )


async def sg_new_assistant_models(message: Message) -> None:
    """На шаге моделей текст не уходит в языковую модель."""
    await message.answer(
        "Выберите модели кнопками или нажмите «Готово».\n/cancel — отмена"
    )


async def _refresh_draft_models(
    query: CallbackQuery, state: FSMContext, assistant: AssistAIClient
) -> None:
    """Обновляет текст и кнопки выбора моделей у создаваемой персоны."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    data = await state.get_data()
    await query.answer()
    await _edit_inline(
        message, _draft_models_text(data, assistant), _draft_models_keyboard(data)
    )


async def on_asst_new_text(
    query: CallbackQuery, state: FSMContext, assistant: AssistAIClient
) -> None:
    """Запоминает текстовую модель до нажатия «Готово»."""
    if await state.get_state() != NewAssistantSG.models.state:
        await query.answer("Сначала начните создание персоны.", show_alert=True)
        return
    raw = (query.data or "").removeprefix("asst:new_text:")
    try:
        selected: str | None = AVAILABLE_TEXT_MODELS[int(raw)]
    except (ValueError, IndexError):
        await query.answer("Неизвестная модель", show_alert=True)
        return
    await state.update_data(flow_text_model=selected)
    await _refresh_draft_models(query, state, assistant)


async def on_asst_new_image(
    query: CallbackQuery, state: FSMContext, assistant: AssistAIClient
) -> None:
    """Запоминает модель картинок до нажатия «Готово»."""
    if await state.get_state() != NewAssistantSG.models.state:
        await query.answer("Сначала начните создание персоны.", show_alert=True)
        return
    raw = (query.data or "").removeprefix("asst:new_image:")
    try:
        selected: str | None = AVAILABLE_IMAGE_MODELS[int(raw)]
    except (ValueError, IndexError):
        await query.answer("Неизвестная модель", show_alert=True)
        return
    await state.update_data(flow_image_model=selected)
    await _refresh_draft_models(query, state, assistant)


async def on_asst_new_default(
    query: CallbackQuery, state: FSMContext, assistant: AssistAIClient
) -> None:
    """Сбрасывает обе модели черновика на дефолт клиента."""
    if await state.get_state() != NewAssistantSG.models.state:
        await query.answer("Сначала начните создание персоны.", show_alert=True)
        return
    await state.update_data(flow_text_model=None, flow_image_model=None)
    await _refresh_draft_models(query, state, assistant)


async def on_asst_new_commit(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    topic_storage: TopicHistoryStorage,
    state: FSMContext,
) -> None:
    """Сохраняет персону из шагов мастера и делает её активной."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    if await state.get_state() != NewAssistantSG.models.state:
        await query.answer("Сначала начните создание персоны.", show_alert=True)
        return
    data = await state.get_data()
    name = data.get("flow_name")
    prompt = data.get("flow_prompt")
    edit_id = data.get("flow_edit_id")
    if not isinstance(name, str) or not isinstance(prompt, str):
        await query.answer("Имя 1–50, промпт ≥ 10", show_alert=True)
        await state.set_state(NewAssistantSG.name)
        await message.answer("Введите имя персоны (1–50 символов).\n/cancel — отмена")
        return
    try:
        fields: dict[str, Any] = {
            "name": name,
            "system_prompt": prompt,
            "text_model": _optional_model(data.get("flow_text_model")),
            "image_model": _optional_model(data.get("flow_image_model")),
        }
        if isinstance(edit_id, str) and edit_id:
            fields["id"] = edit_id
        profile = AssistantProfile.model_validate(fields)
    except ValidationError:
        await query.answer("Имя 1–50, промпт ≥ 10", show_alert=True)
        await state.set_state(NewAssistantSG.name)
        await message.answer("Введите имя персоны (1–50 символов).\n/cancel — отмена")
        return
    chat_id = message.chat.id
    try:
        await assistant_storage.save_assistant(chat_id, profile)
        await assistant_storage.set_active_assistant(chat_id, profile.id)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    await _reset_flow(state)
    logger.info("Сохранён ассистент chat_id=%s id=%s", chat_id, profile.id)
    await query.answer("Сохранено")
    await _cleanup_wizard_ui(
        message,
        state,
        assistant_storage,
        topic_storage,
        also_delete=[message.message_id],
    )
    verb = "Обновлён" if isinstance(edit_id, str) and edit_id else "Создан"
    # Меню уже на невидимом якоре: не вешаем его на текст, иначе удаление
    # подтверждения снова спрячет кнопки.
    await message.answer(f"✅ {verb} ассистент «{profile.name}» ({profile.id}).")
    await _open_settings(message, assistant, assistant_storage, state)


async def _persona_id_for_query(
    query: CallbackQuery, message: Message, assistant_storage: AssistantStorage
) -> str | None:
    """Id активной персоны. Ошибку профиля показывает во всплывающем окне."""
    try:
        profile = await _active_profile(message.chat.id, assistant_storage)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return None
    return profile.id


async def _finish_topic_delete(
    chat_id: int,
    assistant_id: str,
    topic_id: str,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
) -> tuple[str, InlineKeyboardMarkup, ReplyKeyboardMarkup]:
    """Удаляет тему. Пустой каталог получает General, меню берёт её имя."""
    await topic_storage.delete_topic(chat_id, assistant_id, topic_id)
    if not await topic_storage.get_active_topic(chat_id, assistant_id):
        await topic_storage.ensure_active(chat_id, assistant_id, "General")
    text, markup = await _topics_screen(chat_id, assistant_id, topic_storage)
    menu = await _menu_for(chat_id, assistant_storage, topic_storage)
    return text, markup, menu


async def on_topics_refresh(
    query: CallbackQuery,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
) -> None:
    """Перерисовывает список тем с первой страницы в том же сообщении."""
    message = query.message
    if not isinstance(message, Message):
        await query.answer()
        return
    assistant_id = await _persona_id_for_query(query, message, assistant_storage)
    if assistant_id is None:
        return
    try:
        topics = await topic_storage.list_topics(message.chat.id, assistant_id)
        active = await topic_storage.get_active_topic(message.chat.id, assistant_id)
    except TopicStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    await query.answer()
    await _edit_or_answer(
        query,
        _topics_text(topics, active),
        _topics_keyboard(topics, active),
    )


async def on_topics_page(
    query: CallbackQuery,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
) -> None:
    """Открывает другую страницу списка тем."""
    message = query.message
    if not isinstance(message, Message):
        await query.answer()
        return
    try:
        page = int((query.data or "").removeprefix("topic_page:"))
    except ValueError:
        await query.answer("Неверная страница", show_alert=True)
        return
    assistant_id = await _persona_id_for_query(query, message, assistant_storage)
    if assistant_id is None:
        return
    try:
        topics = await topic_storage.list_topics(message.chat.id, assistant_id)
        active = await topic_storage.get_active_topic(message.chat.id, assistant_id)
    except TopicStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    await query.answer()
    await _edit_or_answer(
        query,
        _topics_text(topics, active, page),
        _topics_keyboard(topics, active, page),
    )


async def on_topic_delete_ask(
    query: CallbackQuery,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
) -> None:
    """Спрашивает удаление темы в том же сообщении списка."""
    message = query.message
    if not isinstance(message, Message):
        await query.answer()
        return
    topic_id = (query.data or "").removeprefix("topic_del_ask:")
    assistant_id = await _persona_id_for_query(query, message, assistant_storage)
    if assistant_id is None:
        return
    try:
        topics = await topic_storage.list_topics(message.chat.id, assistant_id)
    except TopicStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    name = next((item["name"] for item in topics if item["id"] == topic_id), None)
    if name is None:
        await query.answer("Тема не найдена.", show_alert=True)
        return
    await query.answer()
    await _edit_inline(
        message,
        f"Удалить тему «{name}» со всей историей?",
        _confirm_row("✅ Да, удалить", f"topic_del_yes:{topic_id}", "topic:list"),
    )


async def on_topic_delete_yes(
    query: CallbackQuery,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Удаляет тему и историю, затем обновляет тот же список тем."""
    message = query.message
    if not isinstance(message, Message):
        await query.answer()
        return
    topic_id = (query.data or "").removeprefix("topic_del_yes:")
    chat_id = message.chat.id
    assistant_id = await _persona_id_for_query(query, message, assistant_storage)
    if assistant_id is None:
        return
    try:
        text, markup, menu = await _finish_topic_delete(
            chat_id, assistant_id, topic_id, topic_storage, assistant_storage
        )
    except TopicStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    logger.info("Удалена тема chat_id=%s topic_id=%s", chat_id, topic_id)
    await query.answer("Тема удалена")
    await _edit_inline(message, text, markup)
    await state.update_data({_SCREEN_TOPICS: message.message_id})
    await _apply_reply_menu(message, menu, state)


async def on_topic_list(
    query: CallbackQuery,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Возвращает сообщение к списку тем."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    assistant_id = await _persona_id_for_query(query, message, assistant_storage)
    if assistant_id is None:
        return
    try:
        text, markup = await _topics_screen(
            message.chat.id, assistant_id, topic_storage
        )
    except TopicStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    await _edit_inline(message, text, markup)
    await state.update_data({_SCREEN_TOPICS: message.message_id})
    await query.answer()


async def on_topic_use(
    query: CallbackQuery,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Делает тему активной, убирает экран и обновляет нижнее меню."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    topic_id = (query.data or "").removeprefix("topic:use:")
    chat_id = message.chat.id
    assistant_id = await _persona_id_for_query(query, message, assistant_storage)
    if assistant_id is None:
        return
    try:
        await topic_storage.set_active_topic(chat_id, assistant_id, topic_id)
        menu = await _menu_for(chat_id, assistant_storage, topic_storage)
    except TopicStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    logger.info("Переключение темы chat_id=%s topic_id=%s", chat_id, topic_id)
    await query.answer()
    await _close_screen(message, state, _SCREEN_TOPICS)
    await _apply_reply_menu(message, menu, state)


async def on_topic_clear_ask(
    query: CallbackQuery,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
) -> None:
    """Спрашивает подтверждение очистки истории темы."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    topic_id = (query.data or "").removeprefix("topic:clear_ask:")
    assistant_id = await _persona_id_for_query(query, message, assistant_storage)
    if assistant_id is None:
        return
    try:
        topics = await topic_storage.list_topics(message.chat.id, assistant_id)
    except TopicStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    name = next((item["name"] for item in topics if item["id"] == topic_id), None)
    if name is None:
        await query.answer("Тема не найдена.", show_alert=True)
        return
    await query.answer()
    await _edit_inline(
        message,
        f"Очистить историю темы «{name}»?",
        _confirm_row("✅ Да, очистить", f"topic:clear_yes:{topic_id}", "topic:list"),
    )


async def on_topic_clear_yes(
    query: CallbackQuery,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Очищает историю темы и показывает список."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    topic_id = (query.data or "").removeprefix("topic:clear_yes:")
    chat_id = message.chat.id
    assistant_id = await _persona_id_for_query(query, message, assistant_storage)
    if assistant_id is None:
        return
    try:
        await topic_storage.clear_topic(chat_id, assistant_id, topic_id)
        text, markup = await _topics_screen(chat_id, assistant_id, topic_storage)
    except TopicStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    logger.info("Тема очищена chat_id=%s topic_id=%s", chat_id, topic_id)
    await query.answer("История очищена")
    await _edit_inline(message, text, markup)
    await state.update_data({_SCREEN_TOPICS: message.message_id})


async def on_topic_rename(
    query: CallbackQuery,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Запрашивает новое название темы."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    topic_id = (query.data or "").removeprefix("topic:rename:")
    assistant_id = await _persona_id_for_query(query, message, assistant_storage)
    if assistant_id is None:
        return
    try:
        topics = await topic_storage.list_topics(message.chat.id, assistant_id)
    except TopicStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    name = next((item["name"] for item in topics if item["id"] == topic_id), None)
    if name is None:
        await query.answer("Тема не найдена.", show_alert=True)
        return
    await _drop_flow(state)
    await state.set_state(RenameTopicSG.name)
    await state.update_data(flow_topic_id=topic_id, flow_topic_name=name)
    await query.answer()
    await message.answer(f"Введите новое название для темы «{name}».\n/cancel — отмена")


async def on_topic_del_ask(
    query: CallbackQuery,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
) -> None:
    """Спрашивает подтверждение удаления темы."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    topic_id = (query.data or "").removeprefix("topic:del_ask:")
    assistant_id = await _persona_id_for_query(query, message, assistant_storage)
    if assistant_id is None:
        return
    try:
        topics = await topic_storage.list_topics(message.chat.id, assistant_id)
    except TopicStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    name = next((item["name"] for item in topics if item["id"] == topic_id), None)
    if name is None:
        await query.answer("Тема не найдена.", show_alert=True)
        return
    await query.answer()
    await _edit_inline(
        message,
        f"Удалить тему «{name}»?",
        _confirm_row("✅ Да, удалить", f"topic:del_yes:{topic_id}", "topic:list"),
    )


async def on_topic_del_yes(
    query: CallbackQuery,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Удаляет тему. Если активных не осталось, создаёт тему по умолчанию."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    topic_id = (query.data or "").removeprefix("topic:del_yes:")
    chat_id = message.chat.id
    assistant_id = await _persona_id_for_query(query, message, assistant_storage)
    if assistant_id is None:
        return
    try:
        text, markup, menu = await _finish_topic_delete(
            chat_id, assistant_id, topic_id, topic_storage, assistant_storage
        )
    except TopicStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    logger.info("Удалена тема chat_id=%s topic_id=%s", chat_id, topic_id)
    await query.answer("Тема удалена")
    await _edit_inline(message, text, markup)
    await state.update_data({_SCREEN_TOPICS: message.message_id})
    await _apply_reply_menu(message, menu, state)


async def on_topic_new(query: CallbackQuery, state: FSMContext) -> None:
    """Запрашивает название новой темы."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    await _drop_flow(state)
    await state.set_state(NewTopicSG.name)
    await query.answer()
    await message.answer("Введите название темы (до 100 символов).\n/cancel — отмена")


async def sg_new_topic_name(
    message: Message,
    state: FSMContext,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
) -> None:
    """Создаёт тему с введённым названием и делает её активной."""
    name = (message.text or "").strip()
    if not name or len(name) > 100:
        await message.answer(
            "Введите название темы (до 100 символов).\n/cancel — отмена"
        )
        return
    chat_id = message.chat.id
    try:
        profile = await _active_profile(chat_id, assistant_storage)
        topic_id = await topic_storage.create_topic(chat_id, profile.id, name)
        await topic_storage.set_active_topic(chat_id, profile.id, topic_id)
        menu = await _menu_for(chat_id, assistant_storage, topic_storage)
    except AssistantStorageError:
        await _reply_profile_error(message)
        return
    except TopicStorageError:
        await _reply_history_error(message)
        return
    await _reset_flow(state)
    logger.info("Новая тема chat_id=%s topic_id=%s", chat_id, topic_id)
    await message.answer(
        f"✅ Создана новая тема: '{name}'. Контекст обнулен.",
        reply_markup=menu,
    )


async def sg_rename_topic_name(
    message: Message,
    state: FSMContext,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
) -> None:
    """Записывает новое название выбранной темы."""
    name = (message.text or "").strip()
    if not name or len(name) > 100:
        await message.answer(
            "Введите новое название (до 100 символов).\n/cancel — отмена"
        )
        return
    data = await state.get_data()
    topic_id = data.get("flow_topic_id")
    chat_id = message.chat.id
    if not isinstance(topic_id, str) or not topic_id:
        await _reset_flow(state)
        menu = await _menu_for(chat_id, assistant_storage, topic_storage)
        await message.answer("Тема для переименования не выбрана.", reply_markup=menu)
        return
    try:
        profile = await _active_profile(chat_id, assistant_storage)
        await topic_storage.rename_topic(chat_id, profile.id, topic_id, name)
        menu = await _menu_for(chat_id, assistant_storage, topic_storage)
    except AssistantStorageError:
        await _reply_profile_error(message)
        return
    except TopicStorageError:
        await _reply_history_error(message)
        return
    await _reset_flow(state)
    logger.info("Тема переименована chat_id=%s topic_id=%s", chat_id, topic_id)
    await _apply_reply_menu(message, menu, state)
    await message.answer(f"Тема переименована в «{name}».")
    await _open_topics(message, topic_storage, assistant_storage, state)


async def _refresh_image_screen(
    query: CallbackQuery, state: FSMContext, assistant: AssistAIClient
) -> None:
    """Обновляет экран картинки после выбора модели или размера."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    data = await state.get_data()
    selected = _optional_model(data.get("image_model"))
    model_id = selected or assistant.config.default_image_model
    size = data.get("image_size")
    allowed = _size_choices(model_id)
    if not isinstance(size, str) or size not in allowed:
        size = _default_image_size(model_id)
        await state.update_data(image_size=size)
    await state.set_state(ImageSG.config)
    await query.answer()
    text, markup = _image_view(selected, size, assistant)
    await _edit_inline(message, text, markup)
    await state.update_data({_SCREEN_IMAGE: message.message_id})


async def on_image_model(
    query: CallbackQuery, state: FSMContext, assistant: AssistAIClient
) -> None:
    """Меняет модель картинки на экране генерации, не записывая её в персону."""
    raw = (query.data or "").removeprefix("image:model:")
    if raw == "default":
        selected: str | None = None
    else:
        try:
            selected = AVAILABLE_IMAGE_MODELS[int(raw)]
        except (ValueError, IndexError):
            await query.answer("Неизвестная модель", show_alert=True)
            return
    data = await state.get_data()
    old_model = (
        _optional_model(data.get("image_model")) or assistant.config.default_image_model
    )
    new_model = selected or assistant.config.default_image_model
    fields: dict[str, Any] = {"image_model": selected}
    if image_model_family(old_model) != image_model_family(new_model):
        fields["image_size"] = _default_image_size(new_model)
    await state.update_data(fields)
    await _refresh_image_screen(query, state, assistant)


async def on_image_size(
    query: CallbackQuery, state: FSMContext, assistant: AssistAIClient
) -> None:
    """Меняет размер картинки на экране генерации."""
    size = (query.data or "").removeprefix("image:size:")
    data = await state.get_data()
    selected = _optional_model(data.get("image_model"))
    model_id = selected or assistant.config.default_image_model
    if size not in _size_choices(model_id):
        await query.answer("Этот размер не подходит модели", show_alert=True)
        return
    await state.update_data(image_size=size)
    await _refresh_image_screen(query, state, assistant)


async def on_image_go(query: CallbackQuery, state: FSMContext) -> None:
    """Переходит к вводу описания картинки."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    await state.set_state(ImageSG.prompt)
    await query.answer()
    await message.answer("Опишите изображение.\n/cancel — отмена")


async def sg_image_prompt(
    message: Message,
    state: FSMContext,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    topic_storage: TopicHistoryStorage,
) -> None:
    """Генерирует картинку по описанию и выбору с экрана."""
    prompt = (message.text or "").strip()
    if not prompt:
        await message.answer("Опишите изображение.\n/cancel — отмена")
        return
    if not security.validate_prompt_length(prompt):
        await message.answer("⚠️ Описание слишком длинное. Максимум 3000 символов.")
        return
    user_id = message.from_user.id if message.from_user else message.chat.id
    if not await rate_limiter.check_image_limit(user_id):
        retry_after = await rate_limiter.get_retry_after(user_id, "image")
        await message.answer(
            f"⏳ Лимит генерации изображений. Подождите {int(retry_after) or 60} секунд."
        )
        logger.warning("rate_limit_exceeded", user_id=user_id, operation="image")
        return
    data = await state.get_data()
    model = _optional_model(data.get("image_model"))
    model_id = model or assistant.config.default_image_model
    size = data.get("image_size")
    if not isinstance(size, str) or size not in _size_choices(model_id):
        size = _default_image_size(model_id)
    await _reset_flow(state)
    menu = await _menu_for(message.chat.id, assistant_storage, topic_storage)
    await _generate_and_send(
        message,
        assistant,
        security.sanitize_prompt(prompt),
        model,
        size,
        reply_markup=menu,
    )


async def on_set_text_menu(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Открывает выбор текстовой модели отдельным сообщением."""
    await _open_model_menu(query, assistant, assistant_storage, state, kind="text")


async def on_set_image_menu(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Открывает выбор модели картинок отдельным сообщением."""
    await _open_model_menu(query, assistant, assistant_storage, state, kind="image")


async def on_settings_open(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Возвращает сообщение к настройкам активной персоны."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    try:
        profile = await _active_profile(message.chat.id, assistant_storage)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    await query.answer()
    text, markup = _settings_view(profile, assistant)
    await _edit_inline(message, text, markup)
    await state.update_data({_SCREEN_SETTINGS: message.message_id})


async def on_settings_close(query: CallbackQuery, state: FSMContext) -> None:
    """Убирает настройки и открытый список моделей."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    await query.answer()
    data = await state.get_data()
    picker_id = data.get(_SCREEN_PICKER)
    bot = message.bot
    if (
        isinstance(picker_id, int)
        and bot is not None
        and picker_id != message.message_id
    ):
        try:
            await bot.delete_message(chat_id=message.chat.id, message_id=picker_id)
        except TelegramBadRequest as exc:
            logger.warning("Не удалось закрыть выбор модели: %s", exc)
        data.pop(_SCREEN_PICKER, None)
        await state.set_data(data)
    await _close_screen(message, state, _SCREEN_SETTINGS)


async def on_set_persona_size(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
) -> None:
    """Записывает размер картинки активной персоны (экран выбора модели)."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    size = (query.data or "").removeprefix("set_size:")
    try:
        profile = await _active_profile(message.chat.id, assistant_storage)
        model_id = _image_model_of(profile, assistant)
        if size not in _size_choices(model_id):
            await query.answer("Этот размер не подходит модели", show_alert=True)
            return
        updated = profile.model_copy(update={"image_size": size})
        await assistant_storage.save_assistant(message.chat.id, updated)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    await query.answer("Размер обновлён")
    text, markup = _model_picker_parts(updated, assistant, "image")
    await _edit_or_answer(query, text, markup)


async def on_picker_done(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Закрывает выбор модели картинок и возвращает к настройкам."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    try:
        profile = await _active_profile(message.chat.id, assistant_storage)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    await query.answer()
    await _return_to_settings(message, state, profile, assistant)


async def on_voice_menu(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Открывает экран голосового ответа поверх сообщения настроек."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    try:
        profile = await _active_profile(message.chat.id, assistant_storage)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    await query.answer()
    text, markup = _voice_view(profile, assistant)
    await _edit_inline(message, text, markup)
    await state.update_data({_SCREEN_SETTINGS: message.message_id})


async def on_voice_mode(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
) -> None:
    """Переключает voice_mode активной персоны."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    mode = (query.data or "").removeprefix("voice_mode:")
    if mode not in {"off", "auto", "always"}:
        await query.answer("Неизвестный режим", show_alert=True)
        return
    try:
        profile = await _active_profile(message.chat.id, assistant_storage)
        updated = profile.model_copy(update={"voice_mode": mode})
        await assistant_storage.save_assistant(message.chat.id, updated)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    label = _VOICE_MODE_LABELS[mode]
    await query.answer(f"Режим: {label}")
    text, markup = _voice_view(updated, assistant)
    await _edit_or_answer(query, text, markup)


async def on_voice_name(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
) -> None:
    """Выбирает голос TTS активной персоны."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    raw = (query.data or "").removeprefix("voice_name:")
    try:
        voice = TTS_VOICES[int(raw)]
    except (ValueError, IndexError):
        await query.answer("Неизвестный голос", show_alert=True)
        return
    try:
        profile = await _active_profile(message.chat.id, assistant_storage)
        updated = profile.model_copy(update={"tts_voice": voice})
        await assistant_storage.save_assistant(message.chat.id, updated)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    await query.answer(f"Голос: {voice}")
    text, markup = _voice_view(updated, assistant)
    await _edit_or_answer(query, text, markup)


async def on_stt_language(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
) -> None:
    """Задаёт язык распознавания речи для активной персоны."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    label = (query.data or "").removeprefix("stt_lang:")
    mapping = {item[0]: item[1] for item in STT_LANGUAGES}
    if label not in mapping:
        await query.answer("Неизвестный язык", show_alert=True)
        return
    language = mapping[label]
    try:
        profile = await _active_profile(message.chat.id, assistant_storage)
        updated = profile.model_copy(update={"stt_language": language})
        await assistant_storage.save_assistant(message.chat.id, updated)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    await query.answer(f"STT: {label}")
    text, markup = _voice_view(updated, assistant)
    await _edit_or_answer(query, text, markup)


async def _open_model_menu(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    state: FSMContext,
    *,
    kind: str,
) -> None:
    """Открывает выбор модели отдельным сообщением. Настройки остаются выше."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    try:
        profile = await _active_profile(message.chat.id, assistant_storage)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    await state.update_data({_SCREEN_SETTINGS: message.message_id})
    text, markup = _model_picker_parts(profile, assistant, kind)
    await query.answer()
    sent = await message.answer(text, reply_markup=markup)
    await state.update_data({_SCREEN_PICKER: sent.message_id})


class _CommandLeavesWizard(BaseMiddleware):
    """Любая команда, кроме /cancel, выходит из мастера до своего хендлера."""

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        if isinstance(event, Message):
            text = event.text or ""
            state = data.get("state")
            if (
                isinstance(state, FSMContext)
                and text.startswith("/")
                and not text.startswith("/cancel")
            ):
                await _drop_flow(state)
        return await handler(event, data)


def _register_handlers(dispatcher: Dispatcher) -> None:
    """Служебные команды — только /start, /help и /cancel.

    Остальное — кнопки нижнего меню и inline. ``aiogram.filters.Text`` удалён
    в aiogram 3.31; любой некомандный текст ловит ``F.text``.
    """
    dispatcher.message.middleware(_CleanChat())
    dispatcher.callback_query.middleware(_CleanChat())
    dispatcher.message.middleware(_CommandLeavesWizard())
    dispatcher.message.register(cmd_start, CommandStart())
    dispatcher.message.register(cmd_start, Command("help"))
    dispatcher.message.register(cmd_cancel, Command("cancel"))
    dispatcher.message.register(
        on_menu_assistants,
        F.text.startswith(f"{_MENU_ASSISTANTS} (")
        | (F.text == MAIN_MENU_TEXT["assistants"]),
    )
    dispatcher.message.register(
        on_menu_topics,
        F.text.startswith(f"{_MENU_TOPICS} (") | (F.text == MAIN_MENU_TEXT["topics"]),
    )
    dispatcher.message.register(on_menu_image, F.text == MAIN_MENU_TEXT["image"])
    dispatcher.message.register(
        on_menu_settings,
        (F.text == _MENU_SETTINGS) | (F.text == MAIN_MENU_TEXT["settings"]),
    )
    dispatcher.callback_query.register(
        on_assistants_page, F.data.startswith("asst_page:")
    )
    dispatcher.callback_query.register(on_topics_page, F.data.startswith("topic_page:"))
    dispatcher.callback_query.register(
        on_asst_rename_ask, F.data.startswith("asst_rename:")
    )
    dispatcher.callback_query.register(
        on_asst_edit_ask, F.data.startswith("asst_edit:")
    )
    dispatcher.callback_query.register(on_topics_refresh, F.data == "topic_list")
    dispatcher.callback_query.register(
        on_topic_delete_ask, F.data.startswith("topic_del_ask:")
    )
    dispatcher.callback_query.register(
        on_topic_delete_yes, F.data.startswith("topic_del_yes:")
    )
    dispatcher.message.register(cmd_unknown, F.text.startswith("/"))
    dispatcher.message.register(sg_new_assistant_name, NewAssistantSG.name)
    dispatcher.message.register(sg_edit_name, EditAssistantSG.name)
    dispatcher.message.register(sg_edit_prompt, EditAssistantSG.prompt)
    dispatcher.message.register(sg_new_assistant_prompt, NewAssistantSG.prompt)
    dispatcher.message.register(sg_new_assistant_models, NewAssistantSG.models)
    dispatcher.message.register(sg_new_topic_name, NewTopicSG.name)
    dispatcher.message.register(sg_rename_topic_name, RenameTopicSG.name)
    dispatcher.message.register(sg_image_prompt, ImageSG.prompt)
    dispatcher.callback_query.register(on_asst_list, F.data == "asst:list")
    dispatcher.callback_query.register(on_asst_use, F.data.startswith("asst:use:"))
    dispatcher.callback_query.register(on_asst_edit, F.data.startswith("asst:edit:"))
    dispatcher.callback_query.register(
        on_asst_del_ask, F.data.startswith("asst:del_ask:")
    )
    dispatcher.callback_query.register(
        on_asst_del_yes, F.data.startswith("asst:del_yes:")
    )
    dispatcher.callback_query.register(on_asst_new_start, F.data == "asst:new")
    dispatcher.callback_query.register(
        on_asst_new_text, F.data.startswith("asst:new_text:")
    )
    dispatcher.callback_query.register(
        on_asst_new_image, F.data.startswith("asst:new_image:")
    )
    dispatcher.callback_query.register(
        on_asst_new_default, F.data == "asst:new_default"
    )
    dispatcher.callback_query.register(on_asst_new_commit, F.data == "asst:new_commit")
    dispatcher.callback_query.register(on_topic_list, F.data == "topic:list")
    dispatcher.callback_query.register(on_topic_use, F.data.startswith("topic:use:"))
    dispatcher.callback_query.register(
        on_topic_clear_ask, F.data.startswith("topic:clear_ask:")
    )
    dispatcher.callback_query.register(
        on_topic_clear_yes, F.data.startswith("topic:clear_yes:")
    )
    dispatcher.callback_query.register(
        on_topic_rename, F.data.startswith("topic:rename:")
    )
    dispatcher.callback_query.register(
        on_topic_del_ask, F.data.startswith("topic:del_ask:")
    )
    dispatcher.callback_query.register(
        on_topic_del_yes, F.data.startswith("topic:del_yes:")
    )
    dispatcher.callback_query.register(on_topic_new, F.data == "topic:new")
    dispatcher.callback_query.register(
        on_image_model, F.data.startswith("image:model:")
    )
    dispatcher.callback_query.register(on_image_size, F.data.startswith("image:size:"))
    dispatcher.callback_query.register(on_image_go, F.data == "image:go")
    dispatcher.callback_query.register(on_settings_open, F.data == "settings:open")
    dispatcher.callback_query.register(on_settings_close, F.data == "settings:close")
    dispatcher.callback_query.register(on_set_text_menu, F.data == "set_text_menu")
    dispatcher.callback_query.register(on_set_image_menu, F.data == "set_image_menu")
    dispatcher.callback_query.register(on_picker_done, F.data == "picker:done")
    dispatcher.callback_query.register(on_voice_menu, F.data == "voice_menu")
    dispatcher.callback_query.register(
        on_set_persona_size, F.data.startswith("set_size:")
    )
    dispatcher.callback_query.register(on_voice_mode, F.data.startswith("voice_mode:"))
    dispatcher.callback_query.register(on_voice_name, F.data.startswith("voice_name:"))
    dispatcher.callback_query.register(on_stt_language, F.data.startswith("stt_lang:"))
    dispatcher.callback_query.register(
        on_set_text_model, F.data.startswith("set_text:")
    )
    dispatcher.callback_query.register(
        on_set_image_model, F.data.startswith("set_image:")
    )
    dispatcher.message.register(handle_voice, F.voice)
    dispatcher.message.register(handle_audio, F.audio)
    dispatcher.message.register(handle_video_file, F.video | F.video_note)
    dispatcher.message.register(handle_photo, F.photo)
    dispatcher.message.register(handle_document, F.document)
    dispatcher.message.register(handle_text, F.text & ~F.text.startswith("/"))


class _ChromeBot(Bot):
    """Запоминает исходящий текст, чтобы убрать его из чата.

    ``message.answer`` вызывает бота напрямую, минуя ``send_message``.
    Ответ персоны и невидимое сообщение с нижним меню отправляются с
    выключенным учётом и в чате остаются.
    """

    async def __call__(
        self,
        method: TelegramMethod[Any],
        request_timeout: int | None = None,
    ) -> Any:
        result = await super().__call__(method, request_timeout=request_timeout)
        if (
            not _KEEP_OUTGOING.get()
            and isinstance(method, SendMessage)
            and isinstance(result, Message)
        ):
            bucket = _CHROME_BUCKET.get()
            if bucket is not None and result.message_id not in bucket:
                bucket.append(result.message_id)
        return result


async def main() -> None:
    """Запускает long polling. Токен бота читается из TELEGRAM_BOT_TOKEN."""
    load_dotenv(Path(__file__).resolve().parent / ".env")
    load_dotenv()
    init_sentry()
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        logger.error("telegram_token_missing")
        raise SystemExit(1)

    bot = _ChromeBot(token=token, session=AiohttpSession())
    storage = await create_fsm_storage(os.getenv("REDIS_URL", "").strip())
    dispatcher = Dispatcher(storage=storage)
    dispatcher.startup.register(on_startup)
    dispatcher.shutdown.register(on_shutdown)
    _register_handlers(dispatcher)
    await dispatcher.start_polling(bot)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Остановка по Ctrl+C")
