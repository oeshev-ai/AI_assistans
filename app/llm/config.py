"""Конфигурация AssistAIClient из окружения."""

from __future__ import annotations

from typing import Any, Literal, cast

import structlog
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.infra.settings import get_settings
from app.infra.settings import load_project_dotenv as load_dotenv

# Ключи лежат рядом с модулем. Загружаем их до импорта клиентов,
# чтобы те увидели то же окружение, даже если процесс запущен не из каталога проекта.
load_dotenv()

from app.llm.models_config import (
    resolve_image_model,
    resolve_text_model,
    resolve_tts_model,
)
from app.llm.providers import ProviderConfig, ProviderRegistry

logger = structlog.get_logger("AssistAI.config")


def _tts_provider_value(raw: str) -> str:
    """Поставщик озвучки: ``polza``, ``proxyapi`` или ``speechkit``."""
    if raw in {"", "polza"}:
        return "polza"
    if raw in {"speechkit", "proxyapi"}:
        return raw
    logger.warning("tts_provider_invalid", value=raw, fallback="polza")
    return "polza"


class AssistAIConfig(BaseModel):
    """Конфигурация клиента. Ключи не попадают в ``repr``.

    ``from_env`` читает ``PROXYAPI_API_KEY`` (обязателен, fallback),
    ``POLZA_API_KEY`` (обязателен, если любой ``*_PROVIDER=polza`` —
    это дефолт), ``OPENWEATHER_API_KEY`` и ``EXCHANGERATE_API_KEY``.
    Пустой ключ погоды или курса валют не мешает запустить клиент:
    соответствующий инструмент просто не регистрируется.

    TTS: ``TTS_PROVIDER`` — поставщик озвучки, ``polza`` | ``speechkit`` |
    ``proxyapi``. ``polza`` и ``proxyapi`` — ``/audio/speech`` у этого хоста.
    ``speechkit`` — Yandex (``YC_SPEECHKIT_API_KEY``, ``YC_FOLDER_ID``,
    ``SPEECHKIT_VOICE`` / ``ROLE`` / ``SPEED``). Голос и формат каталожного
    TTS — ``TTS_VOICE``, ``TTS_FORMAT``.
    Модели общие для провайдеров: ``TEXT_MODEL``, ``TRANSCRIBE_MODEL``,
    ``TTS_MODEL``, ``IMAGE_MODEL``, ``IMAGE_EDIT_MODEL``, ``VIDEO_MODEL``.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    proxyapi_key: str = Field(min_length=1, repr=False)
    openweather_api_key: str = Field(default="", repr=False)
    exchangerate_api_key: str = Field(default="", repr=False)
    base_url: str = "https://api.proxyapi.ru/v1"
    default_text_model: str = "deepseek/deepseek-v4.1-flash-1017"
    default_image_model: str = "openai/gpt-image-2.6-flare"
    default_transcribe_model: str = "openai/whisper-large-v3"
    default_tts_model: str = "openai/gpt-4o-mini-tts"
    default_tts_voice: str = "alloy"
    default_tts_format: str = "opus"
    default_image_edit_model: str = "openai/gpt-image-2.6-flare"
    image_quality: Literal["low", "medium", "high", "xhigh", "max", "auto"] = "high"
    image_edit_size_mode: Literal["source", "auto", "fixed"] = "source"
    image_input_max_edge: int = Field(default=1536, ge=0)
    # Kept for env/from_env compatibility; resolution reads IMAGE/TEXT/TTS_LEGACY_ALIASES directly.
    legacy_model_aliases: bool = True
    default_video_model: str = "bytedance/seedance-2.0"
    tts_provider: Literal["speechkit", "polza", "proxyapi"] = "polza"
    speechkit_api_key: str = Field(default="", repr=False)
    speechkit_folder_id: str = ""
    speechkit_role: str = "friendly"
    speechkit_speed: float = Field(default=1.0, gt=0)
    speechkit_url: str = "https://tts.api.cloud.yandex.net/tts/v3/utteranceSynthesis"
    timeout: float = Field(default=30.0, gt=0)
    max_tool_iterations: int = Field(default=5, ge=1)
    photo_router_model: str = "deepseek/deepseek-v4.1-flash-1017"
    router_confidence: float = Field(default=0.55, ge=0, le=1)
    router_max_tokens: int = Field(default=600, ge=1)
    # Мультипровайдерность. Пустые routes в from_env → polza primary;
    # ProxyAPI остаётся fallback (image/text/audio/video_provider_fallback).
    # Пустой AssistAIConfig без routes: _build_registry сидит на polza, если
    # polza есть в providers, иначе на proxyapi.
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    routes: dict[str, str] = Field(default_factory=dict)
    image_provider_fallback: str = ""
    text_provider_fallback: str = ""
    audio_provider_fallback: str = ""
    video_provider_fallback: str = ""
    rub_per_usd: float = Field(default=304.0, gt=0)
    polza_total_wait: float = Field(default=240.0, gt=0)
    polza_poll_interval: float = Field(default=4.0, gt=0)

    @field_validator("default_image_model", "default_image_edit_model", mode="after")
    @classmethod
    def _alias_legacy_image_model(cls, value: str) -> str:
        resolved = resolve_image_model(value) or value
        if resolved != value:
            logger.warning("image_model_aliased", source="env", old=value, new=resolved)
        return resolved

    @field_validator("default_text_model", "photo_router_model", mode="after")
    @classmethod
    def _alias_legacy_text_model(cls, value: str) -> str:
        resolved = resolve_text_model(value) or value
        if resolved != value:
            logger.warning("text_model_aliased", source="env", old=value, new=resolved)
        return resolved

    @field_validator("default_tts_model", mode="after")
    @classmethod
    def _alias_legacy_tts_model(cls, value: str) -> str:
        resolved = resolve_tts_model(value) or value
        if resolved != value:
            logger.warning("tts_model_aliased", source="env", old=value, new=resolved)
        return resolved

    @field_validator("base_url")
    @classmethod
    def _normalize_base_url(cls, value: str) -> str:
        from app.llm.providers import require_secure_base_url

        return require_secure_base_url(value)

    @field_validator("tts_provider", mode="before")
    @classmethod
    def _normalize_tts_provider(cls, value: Any) -> str:
        cleaned = str(value or "polza").strip().casefold()
        if cleaned not in {"speechkit", "polza", "proxyapi"}:
            raise ValueError("tts_provider должен быть polza, proxyapi или speechkit")
        return cleaned

    @classmethod
    def from_env(cls) -> AssistAIConfig:
        """Собирает конфигурацию из ``app.infra.settings`` и файла ``.env``."""
        load_dotenv()
        settings = get_settings()
        provider: Literal["speechkit", "polza", "proxyapi"] = cast(
            Literal["speechkit", "polza", "proxyapi"],
            _tts_provider_value(settings.tts_provider),
        )
        from app.infra.envparse import ensure_https_url

        speechkit_url = ensure_https_url(
            settings.speechkit_url,
            name="SPEECHKIT_URL",
            default="https://tts.api.cloud.yandex.net/tts/v3/utteranceSynthesis",
        )
        if provider == "speechkit":
            default_tts_voice = settings.speechkit_voice
            default_tts_format = "opus"
            from app.llm.speechkit_tts import SPEECHKIT_MODEL_ID

            default_tts_model = SPEECHKIT_MODEL_ID
        else:
            from app.llm.media_api import normalize_tts_format

            default_tts_voice = settings.tts_voice
            default_tts_format = normalize_tts_format(settings.tts_format)
            default_tts_model = settings.tts_model
        from app.llm.providers import load_routes

        proxy_key = settings.proxyapi_api_key
        polza_key = settings.polza_api_key
        preview_routes = load_routes()
        polza_ops = [op for op, pid in preview_routes.items() if pid == "polza"]
        if polza_ops and not polza_key:
            raise ValueError(
                "POLZA_API_KEY обязателен: маршруты указывают на polza "
                f"({', '.join(polza_ops)})"
            )
        providers: dict[str, ProviderConfig] = {}
        routes: dict[str, str] = {}
        image_fallback = ""
        text_fallback = ""
        audio_fallback = ""
        video_fallback = ""
        if proxy_key:
            # Без ключа ProxyAPI реестр не строим: ошибку отдаст валидация ключа.
            registry = ProviderRegistry.from_env(
                proxyapi_key=proxy_key,
                polza_key=polza_key or None,
            )
            providers = registry.providers
            routes = registry.routes
            image_fallback = registry.image_fallback
            text_fallback = registry.text_fallback
            audio_fallback = registry.audio_fallback
            video_fallback = registry.video_fallback
        return cls(
            proxyapi_key=proxy_key,
            base_url=settings.proxyapi_base_url,
            providers=providers,
            routes=routes,
            image_provider_fallback=image_fallback,
            text_provider_fallback=text_fallback,
            audio_provider_fallback=audio_fallback,
            video_provider_fallback=video_fallback,
            rub_per_usd=settings.rub_per_usd,
            polza_total_wait=settings.polza_total_wait,
            polza_poll_interval=settings.polza_poll_interval,
            openweather_api_key=settings.openweather_api_key,
            exchangerate_api_key=settings.exchangerate_api_key,
            default_text_model=settings.text_model,
            default_transcribe_model=settings.transcribe_model,
            default_tts_model=default_tts_model,
            default_tts_voice=default_tts_voice,
            default_tts_format=default_tts_format,
            default_image_model=settings.image_model,
            default_image_edit_model=settings.image_edit_model,
            image_quality=cast(
                Literal["low", "medium", "high", "xhigh", "max", "auto"],
                settings.image_quality,
            ),
            image_input_max_edge=settings.image_input_max_edge,
            image_edit_size_mode=cast(
                Literal["source", "auto", "fixed"],
                settings.image_edit_size_mode,
            ),
            legacy_model_aliases=settings.image_legacy_aliases_strict,
            default_video_model=settings.video_model,
            tts_provider=provider,
            speechkit_api_key=settings.speechkit_api_key,
            speechkit_folder_id=settings.speechkit_folder_id,
            speechkit_role=settings.speechkit_role,
            speechkit_speed=settings.speechkit_speed,
            speechkit_url=speechkit_url,
            photo_router_model=settings.photo_router_model,
            router_confidence=settings.photo_router_confidence,
            router_max_tokens=settings.photo_router_max_tokens,
        )
