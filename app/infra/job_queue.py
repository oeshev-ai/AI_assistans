"""Optional arq job queue for heavy ProxyAPI operations.

When ``ARQ_ENABLED=1`` and Redis is available, image generation (and optionally
TTS/STT) run in an ``arq`` worker process. Otherwise calls execute in-process
so local development needs no worker.
"""

from __future__ import annotations

import base64
import json
import os
import uuid
from typing import Any
from urllib.parse import urlparse

import structlog

logger = structlog.get_logger("job_queue")

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
        logger.warning("arq_enabled_but_no_redis", fallback="inline")
        _enabled = False
        return
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
        logger.error("arq_pool_failed", error=str(exc), fallback="inline")
        _pool = None
        _enabled = False


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
    """Minimum bot wait / worker job timeout for Polza poll + margin."""
    from app.infra.envparse import env_float

    polza_wait = env_float("POLZA_TOTAL_WAIT", 240.0)
    return max(180.0, polza_wait + 30.0)


def validate_arq_image_timeout() -> None:
    """Fail if ``ARQ_JOB_TIMEOUT`` cannot cover Polza poll + margin.

    A shorter worker timeout kills paid image jobs while the bot may still wait
    (or historically gave up early). Call when the arq queue/worker is enabled.
    """
    from app.infra.envparse import env_float

    arq_timeout = env_float("ARQ_JOB_TIMEOUT", 300.0)
    needed = _polza_image_wait_needed()
    if arq_timeout > 0 and arq_timeout < needed:
        logger.error(
            "arq_image_timeout_below_polza_wait",
            arq_job_timeout=arq_timeout,
            polza_total_wait=env_float("POLZA_TOTAL_WAIT", 240.0),
            needed=needed,
        )
        raise ValueError(
            f"ARQ_JOB_TIMEOUT ({arq_timeout}) must be >= POLZA_TOTAL_WAIT+30 "
            f"({needed})"
        )


def image_result_timeout(explicit: float | None = None) -> float:
    """Wait for queued image jobs at least as long as Polza poll budget.

    Never returns below ``POLZA_TOTAL_WAIT + 30``. Misconfigured
    ``ARQ_JOB_TIMEOUT`` is rejected by ``validate_arq_image_timeout``.
    """
    if explicit is not None:
        return explicit
    return _polza_image_wait_needed()


def _arq_enqueue_failed(task: str) -> Exception:
    """Очередь включена, но job не принят — не уходим в inline (риск OOM бота)."""
    from app.llm.errors import AssistAIError

    logger.error("arq_enqueue_failed_closed", task=task)
    return AssistAIError(
        "Очередь обработки изображений недоступна. Попробуйте позже.",
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
    complexity: str | None = None,
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
            await _pool.set(key, json.dumps(payload), ex=600)
            job = await _pool.enqueue_job(
                "edit_image_task",
                key,
                prompt,
                model,
                labels,
                size,
                user_id,
                complexity,
            )
        except Exception as exc:
            logger.warning("arq_edit_enqueue_failed", error=str(exc))
            try:
                await _pool.delete(key)
            except Exception:
                pass
            raise _arq_enqueue_failed("edit_image_task") from exc
        if job is None:
            try:
                await _pool.delete(key)
            except Exception:
                pass
            raise _arq_enqueue_failed("edit_image_task")
        logger.info("arq_job_enqueued", task="edit_image_task", job_id=job.job_id)
        try:
            result = await job.result(timeout=wait)
            return list(result)
        except Exception as exc:
            # Job was accepted; do not re-run inline (non-idempotent / paid).
            logger.warning("arq_edit_failed", error=str(exc), job_id=job.job_id)
            try:
                await _pool.delete(key)
            except Exception:
                pass
            raise
    return await assistant.edit_images(
        files,
        prompt,
        model=model,
        labels=labels,
        size=size,
        complexity=complexity,
    )


async def synthesize_speech(
    assistant: Any,
    text: str,
    *,
    model: str | None = None,
    voice: str | None = None,
    response_format: str | None = None,
    role: str | None = None,
    timeout: float = 120.0,
    billing_user_id: int | None = None,
) -> Any:
    """Run TTS via arq or inline. Returns ``SpeechResult``-like object."""
    user_id = _billing_user_id(billing_user_id)
    if is_queue_enabled():
        assert _pool is not None
        job = await _pool.enqueue_job(
            "synthesize_speech_task",
            text,
            model,
            voice,
            response_format,
            role,
            user_id,
        )
        if job is None:
            return await assistant.synthesize_speech(
                text,
                model=model,
                voice=voice,
                response_format=response_format,
                role=role,
            )
        logger.info(
            "arq_job_enqueued", task="synthesize_speech_task", job_id=job.job_id
        )
        payload = await job.result(timeout=timeout)
        from app.llm.media_api import SpeechResult

        return SpeechResult(
            audio=base64.b64decode(payload["audio_b64"]),
            filename=payload["filename"],
            mime_type=payload["mime_type"],
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
    timeout: float = 120.0,
    billing_user_id: int | None = None,
) -> str:
    """Run STT via arq (bytes staged in Redis) or inline."""
    user_id = _billing_user_id(billing_user_id)
    if is_queue_enabled() and _pool is not None:
        key = f"arq:stt:{uuid.uuid4().hex}"
        try:
            await _pool.set(key, audio_bytes, ex=600)
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
            except Exception:
                pass
            return await assistant.transcribe_audio(
                audio_bytes,
                filename=filename,
                model=model,
                language=language,
            )
        if job is None:
            await _pool.delete(key)
            return await assistant.transcribe_audio(
                audio_bytes,
                filename=filename,
                model=model,
                language=language,
            )
        logger.info("arq_job_enqueued", task="transcribe_audio_task", job_id=job.job_id)
        try:
            return str(await job.result(timeout=timeout))
        except Exception as exc:
            # Job accepted — do not duplicate STT inline after timeout/failure.
            logger.warning("arq_stt_failed", error=str(exc), job_id=job.job_id)
            try:
                await _pool.delete(key)
            except Exception:
                pass
            raise
    return await assistant.transcribe_audio(
        audio_bytes,
        filename=filename,
        model=model,
        language=language,
    )
