"""arq worker for heavy media jobs (images / TTS / STT).

Run (Docker Compose / repo root shim ``worker.py``)::

    arq worker.WorkerSettings

Equivalent package path::

    arq app.infra.worker.WorkerSettings

Needs ``REDIS_URL`` and provider keys from ``.env`` (``POLZA_API_KEY`` and
``PROXYAPI_API_KEY`` with the default Polza + fallback layout). Enable with
``ARQ_ENABLED=1`` and ``docker compose --profile arq up -d --build``.
"""

from __future__ import annotations

import base64
import json
from typing import Any, ClassVar
from urllib.parse import unquote, urlparse

import structlog

from app.infra.access import is_production, require_redis_password
from app.infra.logging_setup import configure_logging
from app.infra.settings import (
    get_settings,
    load_project_dotenv,
    require_valid_settings,
)
from app.llm.assist_ai import AssistAIClient, AssistAIConfig

logger = structlog.get_logger("worker")


def _warn_polza_user_hash_secret() -> None:
    if not is_production():
        return
    if get_settings().polza_user_hash_secret:
        return
    logger.error(
        "polza_user_hash_secret_missing",
        hint="Set POLZA_USER_HASH_SECRET in .env for HMAC aliases to Polza",
    )


def redis_settings() -> Any:
    from arq.connections import RedisSettings

    redis_url = get_settings().queue_redis_url
    parsed = urlparse(redis_url)
    return RedisSettings(
        host=parsed.hostname or "localhost",
        port=parsed.port or 6379,
        database=int((parsed.path or "/0").lstrip("/") or 0),
        password=unquote(parsed.password) if parsed.password else None,
        username=unquote(parsed.username) if parsed.username else None,
    )


async def attach_worker_billing_redis(redis_url: str) -> Any:
    """Attach shared cost Redis; fail closed in production.

    Returns the Redis client, or ``None`` outside production when URL is empty /
    unreachable (in-memory cost tracker).
    """
    from redis.asyncio import Redis

    from app.infra.access import is_production
    from app.infra.cost_tracker import cost_tracker

    if not redis_url:
        if is_production():
            logger.error("worker_billing_redis_required_in_production")
            raise SystemExit(1)
        logger.warning("worker_cost_redis_skipped", reason="REDIS_URL not set")
        return None
    try:
        billing_redis = Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=2.0,
            socket_timeout=2.0,
        )
        await billing_redis.ping()
        cost_tracker.attach_redis(billing_redis)
        return billing_redis
    except Exception as exc:
        if is_production():
            logger.error("worker_cost_redis_failed_fatal", error=str(exc))
            raise SystemExit(1) from exc
        logger.warning("worker_cost_redis_failed", error=str(exc))
        return None


def _prepare_worker_process() -> None:
    """``.env`` и JSON-логи — при старте воркера, не при импорте модуля."""
    load_project_dotenv()
    configure_logging()


async def startup(ctx: dict[str, Any]) -> None:
    from app.infra.cost_tracker import cost_tracker
    from app.infra.job_queue import validate_arq_image_timeout

    _prepare_worker_process()
    require_valid_settings()
    require_redis_password()

    try:
        validate_arq_image_timeout()
    except ValueError as exc:
        logger.error("arq_timeout_misconfigured", error=str(exc))
        raise SystemExit(1) from exc

    _warn_polza_user_hash_secret()
    ctx["assistant"] = AssistAIClient(AssistAIConfig.from_env())
    cost_tracker.require_llm_budgets_configured()
    redis_url = get_settings().redis_url
    billing_redis = await attach_worker_billing_redis(redis_url)
    ctx["billing_redis"] = billing_redis
    logger.info("worker_started")


async def shutdown(ctx: dict[str, Any]) -> None:
    assistant = ctx.get("assistant")
    if isinstance(assistant, AssistAIClient):
        await assistant.close()
    billing_redis = ctx.get("billing_redis")
    if billing_redis is not None:
        await billing_redis.aclose()
    logger.info("worker_stopped")


def _bind_billing_user(billing_user_id: int | None) -> None:
    from app.infra.cost_tracker import set_billing_user

    set_billing_user(billing_user_id)


async def generate_image_task(
    ctx: dict[str, Any],
    prompt: str,
    model: str | None,
    size: str,
    billing_user_id: int | None = None,
    background: str | None = None,
) -> list[str]:
    assistant: AssistAIClient = ctx["assistant"]
    _bind_billing_user(billing_user_id)
    logger.info(
        "job_generate_image",
        model=model,
        size=size,
        background=background,
        prompt_chars=len(prompt),
        billing_user_id=billing_user_id,
    )
    return await assistant.generate_image(
        prompt=prompt, model=model, size=size, background=background
    )


async def synthesize_speech_task(
    ctx: dict[str, Any],
    text: str,
    model: str | None,
    voice: str | None,
    response_format: str | None,
    role: str | None = None,
    billing_user_id: int | None = None,
) -> dict[str, Any]:
    assistant: AssistAIClient = ctx["assistant"]
    _bind_billing_user(billing_user_id)
    logger.info(
        "job_synthesize_speech",
        chars=len(text),
        voice=voice,
        role=role,
        billing_user_id=billing_user_id,
    )
    speech = await assistant.synthesize_speech(
        text,
        model=model,
        voice=voice,
        response_format=response_format,
        role=role,
    )
    return {
        "audio_b64": base64.b64encode(speech.audio).decode("ascii"),
        "filename": speech.filename,
        "mime_type": speech.mime_type,
        "cost_usd": speech.cost_usd,
        "cost_rub": speech.cost_rub,
        "cost_source": speech.cost_source,
        "provider_id": speech.provider_id,
        "model": speech.model,
    }


async def edit_image_task(
    ctx: dict[str, Any],
    redis_key: str,
    prompt: str,
    model: str | None,
    labels: list[str] | None,
    size: str | None = None,
    billing_user_id: int | None = None,
) -> list[str]:
    """Достаёт файлы правки из Redis и вызывает ``edit_images``.

    Payload удаляется только после успеха. ``max_tries=1``: падение воркера
    не запускает вторую платную задачу. Ключ доживает до TTL и потом исчезает.
    """
    assistant: AssistAIClient = ctx["assistant"]
    _bind_billing_user(billing_user_id)
    redis = ctx["redis"]
    raw = await redis.get(redis_key)
    if not raw:
        raise ValueError("edit payload missing or expired in Redis")
    text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
    payload = json.loads(text)
    if not isinstance(payload, list):
        raise ValueError("edit payload is not a list")
    files: list[tuple[bytes, str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        encoded = item.get("b64")
        if not isinstance(encoded, str) or not encoded:
            continue
        files.append(
            (
                base64.b64decode(encoded),
                str(item.get("filename") or "image.png"),
                str(item.get("mime") or "image/png"),
            )
        )
    logger.info(
        "job_edit_image",
        files=len(files),
        model=model,
        billing_user_id=billing_user_id,
    )
    result = await assistant.edit_images(
        files,
        prompt,
        model=model,
        labels=labels,
        size=size,
    )
    try:
        await redis.delete(redis_key)
    except Exception:
        logger.warning("edit_payload_delete_failed", redis_key=redis_key)
    return result


async def transcribe_audio_task(
    ctx: dict[str, Any],
    redis_key: str,
    filename: str,
    model: str | None,
    language: str | None,
    billing_user_id: int | None = None,
) -> dict[str, Any]:
    """STT из Redis; ключ удаляется только после успеха (crash-retry / TTL)."""
    assistant: AssistAIClient = ctx["assistant"]
    _bind_billing_user(billing_user_id)
    redis = ctx["redis"]
    raw = await redis.get(redis_key)
    if not raw:
        raise ValueError("STT payload missing or expired in Redis")
    audio_bytes = raw if isinstance(raw, (bytes, bytearray)) else bytes(raw)
    logger.info(
        "job_transcribe_audio",
        bytes=len(audio_bytes),
        filename=filename,
        billing_user_id=billing_user_id,
    )
    recognized = await assistant.transcribe_audio(
        bytes(audio_bytes),
        filename=filename,
        model=model,
        language=language,
    )
    try:
        await redis.delete(redis_key)
    except Exception:
        logger.warning("stt_payload_delete_failed", redis_key=redis_key)
    return {
        "text": recognized.text,
        "cost_usd": recognized.cost_usd,
        "cost_rub": recognized.cost_rub,
        "cost_source": recognized.cost_source,
        "provider_id": recognized.provider_id,
        "model": recognized.model,
    }


def _worker_settings_class() -> type:
    _prepare_worker_process()
    require_valid_settings()

    class WorkerSettings:
        """Settings consumed by ``arq worker.WorkerSettings``."""

        functions: ClassVar[list] = [
            generate_image_task,
            edit_image_task,
            synthesize_speech_task,
            transcribe_audio_task,
        ]
        on_startup = startup
        on_shutdown = shutdown
        redis_settings = redis_settings()
        max_jobs = get_settings().arq_max_jobs
        # Одна попытка: повтор после падения создал бы вторую платную задачу.
        max_tries = 1
        job_timeout = get_settings().arq_job_timeout
        # Дефолт arq — час. Картинка и TTS уже прочитаны ботом, в Redis им не место.
        keep_result = 60

    return WorkerSettings


def __getattr__(name: str) -> Any:
    """arq импортирует класс после старта процесса, тесты — нет."""
    if name != "WorkerSettings":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    cls = _worker_settings_class()
    globals()["WorkerSettings"] = cls
    return cls
