"""Конфигурация AssistAIClient из окружения."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, cast

import structlog
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Ключи лежат рядом с модулем. Загружаем их до импорта клиентов,
# чтобы те увидели то же окружение, даже если процесс запущен не из каталога проекта.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")
load_dotenv()

from app.llm.models_config import (
    resolve_image_model,
    resolve_text_model,
    resolve_tts_model,
)
from app.llm.providers import ProviderConfig, ProviderRegistry

logger = structlog.get_logger("AssistAI.config")


def _bounded_float(name: str, default: float) -> float:
    """Читает float из окружения и ограничивает его диапазоном 0..1."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return min(1.0, max(0.0, value))


def _choice(name: str, default: str, allowed: set[str]) -> str:
    """Читает значение из окружения и сверяет его с допустимым набором."""
    raw = os.getenv(name, "").strip().casefold()
    if not raw:
        return default
    if raw not in allowed:
        raise ValueError(f"{name}={raw} не входит в {sorted(allowed)}")
    return raw


def _env_flag(name: str, default: bool) -> bool:
    """Читает 0/1. Пустая переменная оставляет дефолт."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    cleaned = raw.strip().casefold()
    if cleaned in {"1", "true", "yes", "on"}:
        return True
    if cleaned in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} должен быть 0 или 1")


def _non_negative_int(name: str, default: int) -> int:
    """Читает целое ≥ 0. Пустое или битое значение оставляет дефолт."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= 0 else default


def _positive_int(name: str, default: int) -> int:
    """Читает положительное целое из окружения."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _model_env(name: str, default: str) -> str:
    """Значение из окружения. Пустая строка оставляет дефолт."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip() or default


def _positive_float(name: str, default: float) -> float:
    """Читает положительный float. Пустое или битое значение оставляет дефолт."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


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
    default_text_model: str = "deepseek/deepseek-v4.1-flash"
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
    photo_router_model: str = "deepseek/deepseek-v4.1-flash"
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
        """Собирает конфигурацию из переменных окружения и файла ``.env``."""
        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
        load_dotenv()
        provider_raw = os.getenv("TTS_PROVIDER", "").strip().casefold()
        provider: Literal["speechkit", "polza", "proxyapi"] = cast(
            Literal["speechkit", "polza", "proxyapi"],
            _tts_provider_value(provider_raw),
        )
        speechkit_key = os.getenv("YC_SPEECHKIT_API_KEY", "").strip()
        speechkit_voice = os.getenv("SPEECHKIT_VOICE", "marina").strip() or "marina"
        speechkit_role = os.getenv("SPEECHKIT_ROLE", "friendly").strip()
        try:
            speechkit_speed = float(
                os.getenv("SPEECHKIT_SPEED", "1.0").strip() or "1.0"
            )
        except ValueError:
            speechkit_speed = 1.0
        from app.infra.envparse import ensure_https_url

        speechkit_url = ensure_https_url(
            os.getenv("SPEECHKIT_URL", "").strip(),
            name="SPEECHKIT_URL",
            default="https://tts.api.cloud.yandex.net/tts/v3/utteranceSynthesis",
        )
        if provider == "speechkit":
            default_tts_voice = speechkit_voice
            default_tts_format = "opus"
            from app.llm.speechkit_tts import SPEECHKIT_MODEL_ID

            default_tts_model = SPEECHKIT_MODEL_ID
        else:
            from app.llm.media_api import normalize_tts_format

            default_tts_voice = _model_env("TTS_VOICE", "alloy")
            default_tts_format = normalize_tts_format(_model_env("TTS_FORMAT", "opus"))
            default_tts_model = _model_env("TTS_MODEL", "openai/gpt-4o-mini-tts")
        from app.llm.providers import load_routes

        proxy_key = os.getenv("PROXYAPI_API_KEY", "").strip()
        polza_key = os.getenv("POLZA_API_KEY", "").strip()
        base_url = (
            os.getenv("PROXYAPI_BASE_URL", "").strip() or "https://api.proxyapi.ru/v1"
        )
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
        default_text_model = _model_env("TEXT_MODEL", "deepseek/deepseek-v4.1-flash")
        return cls(
            proxyapi_key=os.getenv("PROXYAPI_API_KEY", ""),
            base_url=base_url,
            providers=providers,
            routes=routes,
            image_provider_fallback=image_fallback,
            text_provider_fallback=text_fallback,
            audio_provider_fallback=audio_fallback,
            video_provider_fallback=video_fallback,
            rub_per_usd=_positive_float("RUB_PER_USD", 304.0),
            polza_total_wait=_positive_float("POLZA_TOTAL_WAIT", 240.0),
            polza_poll_interval=_positive_float("POLZA_POLL_INTERVAL", 4.0),
            openweather_api_key=os.getenv("OPENWEATHER_API_KEY", ""),
            exchangerate_api_key=os.getenv("EXCHANGERATE_API_KEY", ""),
            default_text_model=default_text_model,
            default_transcribe_model=_model_env(
                "TRANSCRIBE_MODEL", "openai/whisper-large-v3"
            ),
            default_tts_model=default_tts_model,
            default_tts_voice=default_tts_voice,
            default_tts_format=default_tts_format,
            default_image_model=_model_env("IMAGE_MODEL", "openai/gpt-image-2.6-flare"),
            default_image_edit_model=_model_env(
                "IMAGE_EDIT_MODEL", "openai/gpt-image-2.6-flare"
            ),
            image_quality=cast(
                Literal["low", "medium", "high", "xhigh", "max", "auto"],
                _choice(
                    "IMAGE_QUALITY",
                    "high",
                    {"low", "medium", "high", "xhigh", "max", "auto"},
                ),
            ),
            image_input_max_edge=_non_negative_int("IMAGE_INPUT_MAX_EDGE", 1536),
            image_edit_size_mode=cast(
                Literal["source", "auto", "fixed"],
                _choice(
                    "IMAGE_EDIT_SIZE_MODE",
                    "source",
                    {"source", "auto", "fixed"},
                ),
            ),
            legacy_model_aliases=_env_flag("IMAGE_LEGACY_ALIASES", True),
            default_video_model=_model_env("VIDEO_MODEL", "bytedance/seedance-2.0"),
            tts_provider=provider,
            speechkit_api_key=speechkit_key,
            speechkit_folder_id=os.getenv("YC_FOLDER_ID", "").strip(),
            speechkit_role=speechkit_role,
            speechkit_speed=speechkit_speed,
            speechkit_url=speechkit_url,
            photo_router_model=os.getenv(
                "PHOTO_ROUTER_MODEL", "deepseek/deepseek-v4.1-flash"
            ).strip()
            or "deepseek/deepseek-v4.1-flash",
            router_confidence=_bounded_float("PHOTO_ROUTER_CONFIDENCE", 0.55),
            router_max_tokens=_positive_int("PHOTO_ROUTER_MAX_TOKENS", 600),
        )
