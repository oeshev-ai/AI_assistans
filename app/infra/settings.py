"""Единое чтение переменных окружения приложения.

Имена, значения по умолчанию и разбор живут здесь. Свойства читают
``os.environ`` в момент обращения, поэтому ``monkeypatch.setenv`` в тестах
по-прежнему виден. ``load_dotenv`` не перезаписывает уже заданные переменные.

``REDIS_MAXMEMORY`` и ``APP_IMAGE`` читает только Compose, их здесь нет.
``REDIS_PASSWORD`` нужен процессу: в production пустое значение завершает старт.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_project_dotenv() -> None:
    """Подхватывает ``.env`` проекта, затем ``.env`` из текущего каталога."""
    load_dotenv(_PROJECT_ROOT / ".env")
    load_dotenv()


def getenv(name: str, default: str | None = None) -> str | None:
    """Единственный прямой вызов ``os.getenv`` в приложении."""
    if default is None:
        return os.getenv(name)
    return os.getenv(name, default)


def _stripped(name: str, default: str = "") -> str:
    return (getenv(name, default) or "").strip()


def _choice(name: str, default: str, allowed: set[str]) -> str:
    raw = _stripped(name).casefold()
    if not raw:
        return default
    if raw not in allowed:
        raise ValueError(f"{name}={raw} не входит в {sorted(allowed)}")
    return raw


def _strict_flag(name: str, default: bool) -> bool:
    """0/1. Пустая переменная оставляет дефолт, прочий текст — ошибка."""
    raw = getenv(name)
    if raw is None or not raw.strip():
        return default
    cleaned = raw.strip().casefold()
    if cleaned in {"1", "true", "yes", "on"}:
        return True
    if cleaned in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} должен быть 0 или 1")


def _loose_flag(name: str, default: bool) -> bool:
    """Пусто — дефолт. Иначе истина только для 1/true/yes/on."""
    raw = getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().casefold() in {"1", "true", "yes", "on"}


def _legacy_alias_enabled(name: str) -> bool:
    """Пусто и любой текст кроме 0/false/no оставляют подмену включённой."""
    return _stripped(name, "1").casefold() not in {"0", "false", "no"}


def _model(name: str, default: str) -> str:
    raw = getenv(name)
    if raw is None:
        return default
    return raw.strip() or default


def _bounded_float(name: str, default: float) -> float:
    raw = _stripped(name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return min(1.0, max(0.0, value))


def _non_negative_int(name: str, default: int) -> int:
    raw = _stripped(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= 0 else default


def _positive_int(name: str, default: int) -> int:
    raw = _stripped(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _positive_float(name: str, default: float) -> float:
    raw = _stripped(name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _optional_float(name: str, default: float) -> float:
    """Пустое или нечисловое значение — дефолт. Ноль и отрицательные остаются."""
    from app.infra.envparse import env_float

    return env_float(name, default)


def _optional_int(name: str, default: int) -> int:
    from app.infra.envparse import env_int

    return env_int(name, default)


class Settings:
    """Снимок не кэшируется: каждое свойство читает окружение заново."""

    @property
    def redis_url(self) -> str:
        return _stripped("REDIS_URL")

    @property
    def redis_password(self) -> str:
        return _stripped("REDIS_PASSWORD")

    @property
    def worker_redis_url(self) -> str:
        """arq без ``REDIS_URL`` смотрит на локальный Redis, бот — нет."""
        return _stripped("REDIS_URL", "redis://localhost:6379/0")

    @property
    def topic_ttl_days(self) -> str:
        return getenv("TOPIC_TTL_DAYS", "30") or "30"

    @property
    def max_history_length(self) -> str:
        return getenv("MAX_HISTORY_LENGTH", "20") or "20"

    @property
    def environment(self) -> str:
        return getenv("ENVIRONMENT", "development") or "development"

    @property
    def is_production(self) -> bool:
        return self.environment.strip().lower() == "production"

    @property
    def app_version(self) -> str:
        return getenv("APP_VERSION", "1.0.0") or "1.0.0"

    @property
    def sentry_dsn(self) -> str:
        return _stripped("SENTRY_DSN")

    @property
    def telegram_bot_token(self) -> str:
        return _stripped("TELEGRAM_BOT_TOKEN")

    @property
    def telegram_allowed_user_ids(self) -> str:
        return getenv("TELEGRAM_ALLOWED_USER_IDS", "") or ""

    @property
    def polza_user_hash_secret(self) -> str:
        return _stripped("POLZA_USER_HASH_SECRET")

    @property
    def upload_public_base_url(self) -> str:
        return _stripped("UPLOAD_PUBLIC_BASE_URL")

    @property
    def photo_router_fast_path(self) -> bool:
        return _stripped("PHOTO_ROUTER_FAST_PATH", "0") == "1"

    @property
    def photo_confirm_mode(self) -> str:
        mode = _stripped("PHOTO_CONFIRM_MODE", "A").upper()
        return mode if mode in {"A", "C"} else "A"

    @property
    def photo_max_clarify_rounds(self) -> int:
        return _positive_int("PHOTO_MAX_CLARIFY_ROUNDS", 2)

    @property
    def album_debounce_seconds(self) -> float:
        return _positive_float("ALBUM_DEBOUNCE_SECONDS", 1.2)

    @property
    def max_album_items(self) -> int:
        return _positive_int("MAX_ALBUM_ITEMS", 5)

    @property
    def max_album_total_mb(self) -> int:
        raw = _stripped("MAX_ALBUM_TOTAL_MB", "20")
        try:
            value = int(raw)
        except ValueError:
            value = 20
        return value if value >= 1 else 20

    @property
    def photo_pending_ttl_seconds(self) -> int:
        return _positive_int("PHOTO_PENDING_TTL_SECONDS", 900)

    @property
    def arq_enabled(self) -> bool:
        return _stripped("ARQ_ENABLED").lower() in {"1", "true", "yes", "on"}

    @property
    def health_bind(self) -> str | None:
        return getenv("HEALTH_BIND")

    @property
    def metrics_bind(self) -> str | None:
        return getenv("METRICS_BIND")

    @property
    def metrics_port(self) -> int:
        return _optional_int("METRICS_PORT", 8083)

    @property
    def health_port(self) -> int:
        return _optional_int("HEALTH_PORT", 8082)

    @property
    def upload_max_mb(self) -> int:
        return max(1, _optional_int("UPLOAD_MAX_MB", 512))

    @property
    def upload_max_concurrent(self) -> int:
        return max(1, _optional_int("UPLOAD_MAX_CONCURRENT", 2))

    @property
    def signed_url_expires_in(self) -> int:
        return max(60, _optional_int("SIGNED_URL_EXPIRES_IN", 1800))

    @property
    def arq_max_jobs(self) -> int:
        return _optional_int("ARQ_MAX_JOBS", 10)

    @property
    def arq_job_timeout(self) -> int:
        return _optional_int("ARQ_JOB_TIMEOUT", 450)

    @property
    def arq_job_timeout_seconds(self) -> float:
        return _optional_float("ARQ_JOB_TIMEOUT", 450.0)

    @property
    def openweather_api_key(self) -> str:
        return getenv("OPENWEATHER_API_KEY", "") or ""

    @property
    def exchangerate_api_key(self) -> str:
        return getenv("EXCHANGERATE_API_KEY", "") or ""

    @property
    def proxyapi_api_key(self) -> str:
        return _stripped("PROXYAPI_API_KEY")

    @property
    def polza_api_key(self) -> str:
        return _stripped("POLZA_API_KEY")

    @property
    def proxyapi_base_url(self) -> str:
        return _stripped("PROXYAPI_BASE_URL") or "https://api.proxyapi.ru/v1"

    @property
    def polza_base_url(self) -> str:
        return (
            _stripped("POLZA_BASE_URL", "https://polza.ai/api")
            or "https://polza.ai/api"
        )

    @property
    def polza_chat_base_url(self) -> str:
        return (
            _stripped("POLZA_CHAT_BASE_URL", "https://polza.ai/api/v1")
            or "https://polza.ai/api/v1"
        )

    @property
    def polza_cost_multiplier(self) -> float:
        return _optional_float("POLZA_COST_MULTIPLIER", 0.39)

    @property
    def polza_max_price_image_rub(self) -> float:
        return _optional_float("POLZA_MAX_PRICE_IMAGE_RUB", 12.0)

    @property
    def polza_allow_upstream_fallbacks(self) -> bool:
        return _loose_flag("POLZA_ALLOW_UPSTREAM_FALLBACKS", False)

    @property
    def polza_upstream_only(self) -> tuple[str, ...]:
        raw = _stripped("POLZA_UPSTREAM_ONLY", "OpenAI")
        parts = tuple(part.strip() for part in raw.split(",") if part.strip())
        return parts or ("OpenAI",)

    @property
    def text_provider(self) -> str:
        return _stripped("TEXT_PROVIDER", "polza") or "polza"

    @property
    def image_provider(self) -> str:
        return _stripped("IMAGE_PROVIDER", "polza") or "polza"

    @property
    def audio_provider(self) -> str:
        return _stripped("AUDIO_PROVIDER", "polza") or "polza"

    @property
    def video_provider(self) -> str:
        return _stripped("VIDEO_PROVIDER", "polza") or "polza"

    @property
    def text_provider_fallback(self) -> str:
        return _stripped("TEXT_PROVIDER_FALLBACK", "proxyapi")

    @property
    def image_provider_fallback(self) -> str:
        return _stripped("IMAGE_PROVIDER_FALLBACK", "proxyapi")

    @property
    def audio_provider_fallback(self) -> str:
        return _stripped("AUDIO_PROVIDER_FALLBACK", "proxyapi")

    @property
    def video_provider_fallback(self) -> str:
        return _stripped("VIDEO_PROVIDER_FALLBACK", "proxyapi")

    @property
    def tts_provider(self) -> str:
        return _stripped("TTS_PROVIDER").casefold()

    @property
    def speechkit_api_key(self) -> str:
        return _stripped("YC_SPEECHKIT_API_KEY")

    @property
    def speechkit_folder_id(self) -> str:
        return _stripped("YC_FOLDER_ID")

    @property
    def speechkit_voice(self) -> str:
        return _stripped("SPEECHKIT_VOICE", "marina") or "marina"

    @property
    def speechkit_role(self) -> str:
        return _stripped("SPEECHKIT_ROLE", "friendly")

    @property
    def speechkit_speed(self) -> float:
        raw = _stripped("SPEECHKIT_SPEED", "1.0") or "1.0"
        try:
            return float(raw)
        except ValueError:
            return 1.0

    @property
    def speechkit_url(self) -> str:
        return _stripped("SPEECHKIT_URL")

    @property
    def text_model(self) -> str:
        return _model("TEXT_MODEL", "deepseek/deepseek-v4.1-flash-1017")

    @property
    def transcribe_model(self) -> str:
        return _model("TRANSCRIBE_MODEL", "openai/whisper-large-v3")

    @property
    def tts_model(self) -> str:
        return _model("TTS_MODEL", "openai/gpt-4o-mini-tts")

    @property
    def tts_voice(self) -> str:
        return _model("TTS_VOICE", "alloy")

    @property
    def tts_format(self) -> str:
        return _model("TTS_FORMAT", "opus")

    @property
    def image_model(self) -> str:
        return _model("IMAGE_MODEL", "openai/gpt-image-2.6-flare")

    @property
    def image_edit_model(self) -> str:
        return _model("IMAGE_EDIT_MODEL", "openai/gpt-image-2.6-flare")

    @property
    def image_quality(self) -> str:
        return _choice(
            "IMAGE_QUALITY",
            "high",
            {"low", "medium", "high", "xhigh", "max", "auto"},
        )

    @property
    def image_edit_size_mode(self) -> str:
        return _choice("IMAGE_EDIT_SIZE_MODE", "source", {"source", "auto", "fixed"})

    @property
    def image_input_max_edge(self) -> int:
        return _non_negative_int("IMAGE_INPUT_MAX_EDGE", 1536)

    @property
    def image_legacy_aliases(self) -> bool:
        return _legacy_alias_enabled("IMAGE_LEGACY_ALIASES")

    @property
    def text_legacy_aliases(self) -> bool:
        return _legacy_alias_enabled("TEXT_LEGACY_ALIASES")

    @property
    def tts_legacy_aliases(self) -> bool:
        return _legacy_alias_enabled("TTS_LEGACY_ALIASES")

    @property
    def image_legacy_aliases_strict(self) -> bool:
        return _strict_flag("IMAGE_LEGACY_ALIASES", True)

    @property
    def video_model(self) -> str:
        return _model("VIDEO_MODEL", "bytedance/seedance-2.0")

    @property
    def photo_router_model(self) -> str:
        return _model("PHOTO_ROUTER_MODEL", "deepseek/deepseek-v4.1-flash-1017")

    @property
    def photo_router_confidence(self) -> float:
        return _bounded_float("PHOTO_ROUTER_CONFIDENCE", 0.55)

    @property
    def photo_router_max_tokens(self) -> int:
        return _positive_int("PHOTO_ROUTER_MAX_TOKENS", 600)

    @property
    def rub_per_usd(self) -> float:
        return _positive_float("RUB_PER_USD", 304.0)

    @property
    def cost_rub_per_usd(self) -> float:
        """Та же переменная, но ноль и отрицательные не подменяются дефолтом."""
        return _optional_float("RUB_PER_USD", 304.0)

    @property
    def polza_total_wait(self) -> float:
        return _positive_float("POLZA_TOTAL_WAIT", 240.0)

    @property
    def polza_total_wait_seconds(self) -> float:
        return _optional_float("POLZA_TOTAL_WAIT", 240.0)

    @property
    def polza_poll_interval(self) -> float:
        return _positive_float("POLZA_POLL_INTERVAL", 4.0)

    @property
    def yc_bucket_name(self) -> str:
        return _stripped("YC_BUCKET_NAME")

    @property
    def yc_access_key_id(self) -> str:
        return _stripped("YC_ACCESS_KEY_ID")

    @property
    def yc_secret_access_key(self) -> str:
        return _stripped("YC_SECRET_ACCESS_KEY")

    @property
    def yc_s3_endpoint(self) -> str:
        return _stripped("YC_S3_ENDPOINT")

    @property
    def yc_region(self) -> str:
        return _stripped("YC_REGION", "ru-central1") or "ru-central1"

    @property
    def upload_allowed_mime_types(self) -> str:
        return _stripped("UPLOAD_ALLOWED_MIME_TYPES")

    @property
    def trust_proxy(self) -> bool:
        return _stripped("TRUST_PROXY").lower() in {"1", "true", "yes", "on"}

    @property
    def upload_legacy_presign(self) -> bool:
        return _stripped("UPLOAD_LEGACY_PRESIGN", "0").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    @property
    def cost_chat_input_per_1k(self) -> float:
        return _optional_float("COST_CHAT_INPUT_PER_1K", 0.00015)

    @property
    def cost_chat_output_per_1k(self) -> float:
        return _optional_float("COST_CHAT_OUTPUT_PER_1K", 0.0006)

    @property
    def cost_image_usd(self) -> float:
        return _optional_float("COST_IMAGE_USD", 0.04)

    @property
    def cost_image_flux_usd(self) -> float:
        return _optional_float("COST_IMAGE_FLUX_USD", 0.005)

    @property
    def cost_image_flat_map(self) -> str | None:
        return getenv("COST_IMAGE_FLAT_MAP")

    @property
    def image_out_usd_per_1m(self) -> float:
        return _optional_float("IMAGE_OUT_USD_PER_1M", 30.0)

    @property
    def image_in_usd_per_1m(self) -> float:
        return _optional_float("IMAGE_IN_USD_PER_1M", 8.0)

    @property
    def text_in_usd_per_1m(self) -> float:
        return _optional_float("TEXT_IN_USD_PER_1M", 5.0)

    @property
    def text_out_usd_per_1m(self) -> float:
        return _optional_float("TEXT_OUT_USD_PER_1M", 10.0)

    @property
    def cost_video_usd(self) -> float:
        return _optional_float("COST_VIDEO_USD", 0.20)

    @property
    def cost_stt_per_minute(self) -> float:
        return _optional_float("COST_STT_PER_MINUTE", 0.006)

    @property
    def cost_tts_per_1k_chars(self) -> float:
        return _optional_float("COST_TTS_PER_1K_CHARS", 0.015)

    @property
    def llm_daily_budget_usd(self) -> float:
        return _optional_float("LLM_DAILY_BUDGET_USD", 0.0)

    @property
    def llm_user_daily_budget_usd(self) -> float:
        return _optional_float("LLM_USER_DAILY_BUDGET_USD", 0.0)


_SETTINGS = Settings()


def get_settings() -> Settings:
    """Общий объект настроек. Значения не замораживаются."""
    return _SETTINGS


load_project_dotenv()
