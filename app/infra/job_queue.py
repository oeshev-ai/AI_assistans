"""Optional arq job queue for heavy ProxyAPI operations.

When ``ARQ_ENABLED=1`` and Redis is available, image generation (and optionally
TTS/STT) run in an ``arq`` worker process. Otherwise calls execute in-process
so local development needs no worker.
"""

from __future__ import annotations

import base64
import json
import math
import os
import uuid
from typing import Any
from urllib.parse import urlparse

import structlog

from app.llm.media_api import SpeechResult, TranscriptionResult

logger = structlog.get_logger("job_queue")

# Создание картинки может занять весь read-timeout, затем опрос — POLZA_TOTAL_WAIT.
_IMAGE_CREATE_TIMEOUT = 180.0
_DEFAULT_ARQ_JOB_TIMEOUT = 450.0


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _optional_str(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _coerce_transcription(result: Any) -> TranscriptionResult:
    """Приводит ответ STT к ``TranscriptionResult`` (строка — без цены)."""
    if isinstance(result, TranscriptionResult):
        return result
    if isinstance(result, dict):
        text = result.get("text")
        return TranscriptionResult(
            text=text if isinstance(text, str) else "",
            cost_usd=_optional_float(result.get("cost_usd")),
            cost_rub=_optional_float(result.get("cost_rub")),
            cost_source=_optional_str(result.get("cost_source")),
            provider_id=_optional_str(result.get("provider_id")),
            model=_optional_str(result.get("model")),
        )
    return TranscriptionResult(text="" if result is None else str(result))


_pool: Any | None = None
_enabled = False


def _redis_settings_from_url(redis_url: str) -> Any:
    from arq.connections import RedisSettings

    parsed = urlparse(redis_url)
    # Empty userinfo (redis://:@host) must not become AUTH "".
    password = parsed.password or None
    username = parsed.username or None
    return RedisSettings(
        host=parsed.hostname or "localhost",
        port=parsed.port or 6379,
        database=int((parsed.path or "/0").lstrip("/") or 0),
        password=password,
        username=username,
    )


async def init_job_queue(redis_url: str) -> None:
    """Create an arq pool when queueing is enabled."""
    global _pool, _enabled
    flag = os.getenv("ARQ_ENABLED", "").strip().lower()
    _enabled = flag in {"1", "true", "yes", "on"}
    if not _enabled:
        logger.info("arq_disabled")
        return
    if not redis_url:
        # ARQ_ENABLED=1 without Redis: fail closed (no silent inline OOM risk).
        logger.error("arq_enabled_but_no_redis")
        raise SystemExit(1)
    try:
        validate_arq_image_timeout()
    except ValueError as exc:
        logger.error("arq_timeout_misconfigured", error=str(exc))
        raise SystemExit(1) from exc
    try:
        from arq import create_pool

        _pool = await create_pool(_redis_settings_from_url(redis_url))
        logger.info("arq_pool_ready")
    except SystemExit:
        raise
    except Exception as exc:
        logger.error("arq_pool_failed", error=str(exc))
        _pool = None
        _enabled = False
        raise SystemExit(1) from exc


async def close_job_queue() -> None:
    """Close the arq Redis pool if it was opened."""
    global _pool
    if _pool is not None:
        await _pool.aclose()
        _pool = None
        logger.info("arq_pool_closed")


def is_queue_enabled() -> bool:
    return _enabled and _pool is not None


def _billing_user_id(explicit: int | None = None) -> int | None:
    if explicit is not None:
        return explicit
    from app.infra.cost_tracker import get_billing_user

    return get_billing_user()


def _polza_image_wait_needed() -> float:
    """Минимум ожидания: таймаут создания плюс опрос Polza и запас."""
    from app.infra.envparse import env_float

    polza_wait = max(0.0, env_float("POLZA_TOTAL_WAIT", 240.0))
    return _IMAGE_CREATE_TIMEOUT + polza_wait + 30.0


def validate_arq_image_timeout() -> None:
    """Fail if ``ARQ_JOB_TIMEOUT`` cannot cover Polza poll + margin.

    A shorter worker timeout kills paid image jobs while the bot may still wait
    (or historically gave up early). Call when the arq queue/worker is enabled.
    """
    from app.infra.envparse import env_float

    arq_timeout = env_float("ARQ_JOB_TIMEOUT", _DEFAULT_ARQ_JOB_TIMEOUT)
    needed = _polza_image_wait_needed()
    if arq_timeout > 0 and arq_timeout < needed:
        logger.error(
            "arq_image_timeout_below_polza_wait",
            arq_job_timeout=arq_timeout,
            polza_total_wait=env_float("POLZA_TOTAL_WAIT", 240.0),
            needed=needed,
        )
        raise ValueError(
            "ARQ_JOB_TIMEOUT "
            f"({arq_timeout}) must be >= image create + POLZA_TOTAL_WAIT + 30 ({needed})"
        )


def image_result_timeout(explicit: float | None = None) -> float:
    """Бот ждёт задачу не меньше, чем воркеру разрешено её выполнять.

    Ниже ``image create + POLZA_TOTAL_WAIT + 30`` не опускается.
    Короткий ``ARQ_JOB_TIMEOUT`` отклоняет ``validate_arq_image_timeout``.
    """
    if explicit is not None:
        return explicit
    from app.infra.envparse import env_float

    job_timeout = env_float("ARQ_JOB_TIMEOUT", _DEFAULT_ARQ_JOB_TIMEOUT)
    return max(_polza_image_wait_needed(), job_timeout)


def audio_result_timeout(explicit: float | None = None) -> float:
    """Ожидание TTS/STT совпадает с таймаутом задачи, чтобы бот не сдался раньше."""
    if explicit is not None:
        return explicit
    from app.infra.envparse import env_float

    return max(120.0, env_float("ARQ_JOB_TIMEOUT", _DEFAULT_ARQ_JOB_TIMEOUT))


def _payload_ttl_seconds() -> int:
    """Redis TTL for staged ARQ payloads (edit/STT bytes).

    Must outlive queue delay + worker ``ARQ_JOB_TIMEOUT`` so backlog does not
    expire the body before the job starts.
    """
    from app.infra.envparse import env_float

    job_timeout = max(0.0, env_float("ARQ_JOB_TIMEOUT", _DEFAULT_ARQ_JOB_TIMEOUT))
    polza_wait = max(0.0, env_float("POLZA_TOTAL_WAIT", 240.0))
    return int(max(600.0, job_timeout + 120.0, polza_wait + 120.0))


def _arq_enqueue_failed(task: str) -> Exception:
    """Очередь включена, но job не принят — не уходим в inline (риск OOM бота)."""
    from app.llm.errors import AssistAIError

    logger.error("arq_enqueue_failed_closed", task=task)
    kind = (
        "аудио"
        if "speech" in task or "transcribe" in task or "stt" in task
        else "изображений"
    )
    return AssistAIError(
        f"Очередь обработки {kind} недоступна. Попробуйте позже.",
        error_code="SERVICE_UNAVAILABLE",
    )


async def generate_image(
    assistant: Any,
    prompt: str,
    model: str | None,
    size: str,
    *,
    timeout: float | None = None,
    billing_user_id: int | None = None,
) -> list[str]:
    """Generate an image via arq worker or inline client."""
    user_id = _billing_user_id(billing_user_id)
    wait = image_result_timeout(timeout)
    if is_queue_enabled():
        assert _pool is not None
        try:
            job = await _pool.enqueue_job(
                "generate_image_task",
                prompt,
                model,
                size,
                user_id,
            )
        except Exception as exc:
            logger.warning("arq_generate_enqueue_failed", error=str(exc))
            raise _arq_enqueue_failed("generate_image_task") from exc
        if job is None:
            raise _arq_enqueue_failed("generate_image_task")
        logger.info("arq_job_enqueued", task="generate_image_task", job_id=job.job_id)
        result = await job.result(timeout=wait)
        return list(result)
    return await assistant.generate_image(prompt=prompt, model=model, size=size)


async def edit_images(
    assistant: Any,
    files: list[tuple[bytes, str, str]],
    prompt: str,
    *,
    model: str | None = None,
    labels: list[str] | None = None,
    size: str | None = None,
    timeout: float | None = None,
    billing_user_id: int | None = None,
) -> list[str]:
    """Правит фото через arq (байты в Redis) или сразу в процессе."""
    user_id = _billing_user_id(billing_user_id)
    wait = image_result_timeout(timeout)
    if is_queue_enabled() and _pool is not None:
        key = f"arq:edit:{uuid.uuid4().hex}"
        payload = [
            {
                "b64": base64.b64encode(data).decode("ascii"),
                "filename": name,
                "mime": mime,
            }
            for data, name, mime in files
        ]
        try:
            await _pool.set(key, json.dumps(payload), ex=_payload_ttl_seconds())
            job = await _pool.enqueue_job(
                "edit_image_task",
                key,
                prompt,
                model,
                labels,
                size,
                user_id,
            )
        except Exception as exc:
            logger.warning("arq_edit_enqueue_failed", error=str(exc))
            try:
                await _pool.delete(key)
            except Exception as cleanup_exc:
                logger.debug("arq_edit_payload_cleanup_failed", error=str(cleanup_exc))
            raise _arq_enqueue_failed("edit_image_task") from exc
        if job is None:
            try:
                await _pool.delete(key)
            except Exception as cleanup_exc:
                logger.debug("arq_edit_payload_cleanup_failed", error=str(cleanup_exc))
            raise _arq_enqueue_failed("edit_image_task")
        logger.info("arq_job_enqueued", task="edit_image_task", job_id=job.job_id)
        try:
            result = await job.result(timeout=wait)
            return list(result)
        except Exception as exc:
            # Job accepted — do not re-run inline. Payload stays until TTL.
            logger.warning("arq_edit_failed", error=str(exc), job_id=job.job_id)
            raise
    return await assistant.edit_images(
        files,
        prompt,
        model=model,
        labels=labels,
        size=size,
    )


async def synthesize_speech(
    assistant: Any,
    text: str,
    *,
    model: str | None = None,
    voice: str | None = None,
    response_format: str | None = None,
    role: str | None = None,
    timeout: float | None = None,
    billing_user_id: int | None = None,
) -> Any:
    """Run TTS via arq or inline. Returns ``SpeechResult``-like object."""
    user_id = _billing_user_id(billing_user_id)
    timeout = audio_result_timeout(timeout)
    if is_queue_enabled():
        assert _pool is not None
        try:
            job = await _pool.enqueue_job(
                "synthesize_speech_task",
                text,
                model,
                voice,
                response_format,
                role,
                user_id,
            )
        except Exception as exc:
            logger.warning("arq_tts_enqueue_failed", error=str(exc))
            raise _arq_enqueue_failed("synthesize_speech_task") from exc
        if job is None:
            raise _arq_enqueue_failed("synthesize_speech_task")
        logger.info(
            "arq_job_enqueued", task="synthesize_speech_task", job_id=job.job_id
        )
        payload = await job.result(timeout=timeout)
        return SpeechResult(
            audio=base64.b64decode(payload["audio_b64"]),
            filename=payload["filename"],
            mime_type=payload["mime_type"],
            cost_usd=_optional_float(payload.get("cost_usd")),
            cost_rub=_optional_float(payload.get("cost_rub")),
            cost_source=_optional_str(payload.get("cost_source")),
            provider_id=_optional_str(payload.get("provider_id")),
            model=_optional_str(payload.get("model")),
        )
    return await assistant.synthesize_speech(
        text,
        model=model,
        voice=voice,
        response_format=response_format,
        role=role,
    )


async def transcribe_audio(
    assistant: Any,
    audio_bytes: bytes,
    *,
    filename: str = "voice.ogg",
    model: str | None = None,
    language: str | None = None,
    timeout: float | None = None,
    billing_user_id: int | None = None,
) -> TranscriptionResult:
    """Run STT via arq (bytes staged in Redis) or inline.

    Возвращает ``TranscriptionResult``. Строка от старого воркера или
    тестового двойника приходит без цены.
    """
    user_id = _billing_user_id(billing_user_id)
    timeout = audio_result_timeout(timeout)
    if is_queue_enabled() and _pool is not None:
        key = f"arq:stt:{uuid.uuid4().hex}"
        try:
            await _pool.set(key, audio_bytes, ex=_payload_ttl_seconds())
            job = await _pool.enqueue_job(
                "transcribe_audio_task",
                key,
                filename,
                model,
                language,
                user_id,
            )
        except Exception as exc:
            logger.warning("arq_stt_enqueue_failed", error=str(exc))
            try:
                await _pool.delete(key)
            except Exception as cleanup_exc:
                logger.debug("arq_stt_payload_cleanup_failed", error=str(cleanup_exc))
            raise _arq_enqueue_failed("transcribe_audio_task") from exc
        if job is None:
            try:
                await _pool.delete(key)
            except Exception as cleanup_exc:
                logger.debug("arq_stt_payload_cleanup_failed", error=str(cleanup_exc))
            raise _arq_enqueue_failed("transcribe_audio_task")
        logger.info("arq_job_enqueued", task="transcribe_audio_task", job_id=job.job_id)
        try:
            return _coerce_transcription(await job.result(timeout=timeout))
        except Exception as exc:
            # Job accepted — do not duplicate STT inline. Payload TTL covers cleanup.
            logger.warning("arq_stt_failed", error=str(exc), job_id=job.job_id)
            raise
    return _coerce_transcription(
        await assistant.transcribe_audio(
            audio_bytes,
            filename=filename,
            model=model,
            language=language,
        )
    )
