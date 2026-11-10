"""Optional arq job queue for heavy ProxyAPI operations.

When ``ARQ_ENABLED=1`` and Redis is available, image generation (and optionally
TTS/STT) run in an ``arq`` worker process. Otherwise calls execute in-process
so local development needs no worker.
"""

from __future__ import annotations

import base64
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
    return RedisSettings(
        host=parsed.hostname or "localhost",
        port=parsed.port or 6379,
        database=int((parsed.path or "/0").lstrip("/") or 0),
        password=parsed.password,
        username=parsed.username,
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
        from arq import create_pool

        _pool = await create_pool(_redis_settings_from_url(redis_url))
        logger.info("arq_pool_ready")
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


async def generate_image(
    assistant: Any,
    prompt: str,
    model: str | None,
    size: str,
    *,
    timeout: float = 180.0,
) -> list[str]:
    """Generate an image via arq worker or inline client."""
    if is_queue_enabled():
        assert _pool is not None
        job = await _pool.enqueue_job(
            "generate_image_task",
            prompt,
            model,
            size,
        )
        if job is None:
            logger.warning("arq_enqueue_skipped_inline", task="generate_image_task")
            return await assistant.generate_image(prompt=prompt, model=model, size=size)
        logger.info("arq_job_enqueued", task="generate_image_task", job_id=job.job_id)
        result = await job.result(timeout=timeout)
        return list(result)
    return await assistant.generate_image(prompt=prompt, model=model, size=size)


async def synthesize_speech(
    assistant: Any,
    text: str,
    *,
    model: str | None = None,
    voice: str | None = None,
    response_format: str | None = None,
    timeout: float = 120.0,
) -> Any:
    """Run TTS via arq or inline. Returns ``SpeechResult``-like object."""
    if is_queue_enabled():
        assert _pool is not None
        job = await _pool.enqueue_job(
            "synthesize_speech_task",
            text,
            model,
            voice,
            response_format,
        )
        if job is None:
            return await assistant.synthesize_speech(
                text,
                model=model,
                voice=voice,
                response_format=response_format,
            )
        logger.info(
            "arq_job_enqueued", task="synthesize_speech_task", job_id=job.job_id
        )
        payload = await job.result(timeout=timeout)
        from Assist_ai import SpeechResult

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
    )


async def transcribe_audio(
    assistant: Any,
    audio_bytes: bytes,
    *,
    filename: str = "voice.ogg",
    model: str | None = None,
    language: str | None = None,
    timeout: float = 120.0,
) -> str:
    """Run STT via arq (bytes staged in Redis) or inline."""
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
            )
            if job is None:
                await _pool.delete(key)
                return await assistant.transcribe_audio(
                    audio_bytes,
                    filename=filename,
                    model=model,
                    language=language,
                )
            logger.info(
                "arq_job_enqueued", task="transcribe_audio_task", job_id=job.job_id
            )
            return str(await job.result(timeout=timeout))
        except Exception as exc:
            logger.warning("arq_stt_failed_inline", error=str(exc))
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
