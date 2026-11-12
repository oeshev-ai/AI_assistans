"""Bot package constants, ContextVars and shared imports."""

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

import app.infra.job_queue as job_queue
from app.infra.cost_tracker import BudgetExceededError, cost_tracker, set_billing_user
from app.infra.health_server import start_health_server
from app.infra.logging_setup import (
    bind_request_context,
    clear_request_context,
    configure_logging,
)
from app.infra.metrics import note_active_user, observe_message, start_metrics_server
from app.infra.rate_limiter import rate_limiter
from app.infra.security import security
from app.llm.assist_ai import (
    AssistAIClient,
    AssistAIConfig,
    AssistAIError,
    ImageModelScope,
    image_model_family,
)
from app.llm.models_config import (
    AVAILABLE_IMAGE_MODELS,
    AVAILABLE_TEXT_MODELS,
    AVAILABLE_TTS_MODELS,
    ELEVENLABS_TTS_VOICES,
    OPENAI_TTS_VOICES,
    is_elevenlabs_tts_model,
    model_hint,
)
from app.llm.speechkit_tts import (
    SPEECHKIT_VOICES,
    is_speechkit_tts_model,
    resolve_speechkit_role,
    resolve_speechkit_voice,
    speechkit_roles_for,
)
from app.llm.vision import (
    MAX_IMAGE_SIZE_MB,
    VisionError,
    build_image_content,
    build_multimodal_content,
    download_telegram_file,
    ensure_within_size_limit,
    extract_image_from_message,
    process_pdf,
)
from app.media.media_tools import (
    MediaTurnScope,
    OutboundMedia,
    UserMedia,
    reset_user_media,
    set_user_media,
)
from app.media.media_utils import MediaProcessingError, prepare_audio_for_stt
from app.media.upload_server import (
    is_upload_feature_ready,
    public_upload_url,
    setup_upload_routes,
)
from app.storage.assistant_presets import (
    ASSISTANT_PRESETS,
    AssistantPreset,
    get_preset,
    preset_profile,
)
from app.storage.assistant_storage import (
    DEFAULT_ASSISTANT_ID,
    DEFAULT_ASSISTANT_NAME,
    AssistantProfile,
    AssistantStorage,
    AssistantStorageError,
    compose_system_prompt,
    create_assistant_storage,
)
from app.storage.history_storage import RedisConfig
from app.storage.persona_icons import (
    DEFAULT_ICON,
    PERSONA_ICONS,
    normalize_icon,
    persona_label,
)
from app.storage.s3_storage import S3Storage, S3StorageError, get_storage
from app.storage.topic_history import (
    TopicConfig,
    TopicHistoryStorage,
    TopicStorageError,
    create_topic_storage,
)
from app.storage.upload_sessions import (
    UploadMediaKind,
    UploadRequestContext,
    UploadSession,
    UploadSessionStore,
    reset_upload_request_context,
    set_upload_request_context,
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
    "Файл больше ~20 МБ: бот сам пришлёт ссылку на загрузку в облако "
    "(или напишите «нужна ссылка на загрузку»).\n\n"
    "Внизу кнопки: персоны и темы.\n"
    "Каталог готовых персон — в /settings.\n"
    "Меню (☰): /settings — настройки, /help — справка, /cancel — отмена мастера"
)
InputKind = Literal["text", "voice"]
MAX_VOICE_DURATION_SECONDS = 600
MAX_VOICE_SIZE_MB = 20
MAX_MEDIA_SIZE_MB = 20  # лимит Telegram Bot API getFile
SHOW_RECOGNIZED_TEXT = True
VOICE_RECOGNIZED_PREVIEW_LIMIT = 500
TTS_MAX_CHARS = 3500
# Совместимость со старым именем; актуальный список — ``_tts_voices_for``.
PROXYAPI_TTS_VOICES: tuple[str, ...] = OPENAI_TTS_VOICES
TTS_VOICES: tuple[str, ...] = OPENAI_TTS_VOICES
_MENU_ASSISTANTS = "Персоны"
_MENU_TOPICS = "Темы"
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
    "flow_icon",
    "flow_preset_id",
    "image_model",
    "image_size",
)
_SCREEN_ASSISTANTS = "screen_assistants"
_SCREEN_TOPICS = "screen_topics"
_SCREEN_SETTINGS = "screen_settings"
_SCREEN_PICKER = "screen_picker"
_SCREEN_IMAGE = "screen_image"
_SCREEN_PRESETS = "screen_presets"
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
    _SCREEN_PRESETS,
)
_VOICE_MODE_LABELS: dict[str, str] = {
    "off": "выкл",
    "auto": "авто",
    "always": "всегда",
}
MSG_UNAVAILABLE = "⚠️ Сервис временно недоступен. Попробуйте позже."
MSG_TIMEOUT = "⚠️ Превышено время ожидания."
MSG_INTERNAL = "❌ Произошла внутренняя ошибка."
MSG_IMAGE_FAILED = "Не удалось сгенерировать изображение."
MSG_HISTORY = "⚠️ Временные проблемы с историей диалога. Попробуйте позже."
MSG_PROFILE = "⚠️ Не удалось прочитать профиль ассистента. Попробуйте позже."


from app.bot.states import (
    EditAssistantSG,
    ImageSG,
    NewAssistantSG,
    NewTopicSG,
    RenameTopicSG,
)

# Include private screen/menu keys so consumers can import them explicitly.
__all__ = [name for name in globals() if not name.startswith("__")]
