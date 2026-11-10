"""arq worker process for heavy ProxyAPI jobs.

Run::

    arq worker.WorkerSettings

Requires ``REDIS_URL`` and ``PROXYAPI_API_KEY``.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urlparse

import structlog
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv()

from Assist_ai import AssistAIClient, AssistAIConfig
from logging_setup import configure_logging

configure_logging()
logger = structlog.get_logger("worker")


def redis_settings() -> Any:
    from arq.connections import RedisSettings

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0").strip()
    parsed = urlparse(redis_url)
    return RedisSettings(
        host=parsed.hostname or "localhost",
        port=parsed.port or 6379,
        database=int((parsed.path or "/0").lstrip("/") or 0),
        password=parsed.password,
        username=parsed.username,
    )


async def startup(ctx: dict[str, Any]) -> None:
    from redis.asyncio import Redis

    from cost_tracker import cost_tracker

    ctx["assistant"] = AssistAIClient(AssistAIConfig.from_env())
    redis_url = os.getenv("REDIS_URL", "").strip()
    billing_redis = None
    if redis_url:
        try:
            billing_redis = Redis.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=2.0,
                socket_timeout=2.0,
            )
            await billing_redis.ping()
            cost_tracker.attach_redis(billing_redis)
            ctx["billing_redis"] = billing_redis
        except Exception as exc:  # noqa: BLE001
            logger.warning("worker_cost_redis_failed", error=str(exc))
    logger.info("worker_started")


async def shutdown(ctx: dict[str, Any]) -> None:
    assistant = ctx.get("assistant")
    if isinstance(assistant, AssistAIClient):
        await assistant.close()
    billing_redis = ctx.get("billing_redis")
    if billing_redis is not None:
        await billing_redis.aclose()
    logger.info("worker_stopped")


async def generate_image_task(
    ctx: dict[str, Any],
    prompt: str,
    model: str | None,
    size: str,
) -> list[str]:
    assistant: AssistAIClient = ctx["assistant"]
    logger.info("job_generate_image", model=model, size=size, prompt_chars=len(prompt))
    return await assistant.generate_image(prompt=prompt, model=model, size=size)


async def synthesize_speech_task(
    ctx: dict[str, Any],
    text: str,
    model: str | None,
    voice: str | None,
    response_format: str | None,
) -> dict[str, str]:
    assistant: AssistAIClient = ctx["assistant"]
    logger.info("job_synthesize_speech", chars=len(text), voice=voice)
    speech = await assistant.synthesize_speech(
        text,
        model=model,
        voice=voice,
        response_format=response_format,
    )
    return {
        "audio_b64": base64.b64encode(speech.audio).decode("ascii"),
        "filename": speech.filename,
        "mime_type": speech.mime_type,
    }


async def transcribe_audio_task(
    ctx: dict[str, Any],
    redis_key: str,
    filename: str,
    model: str | None,
    language: str | None,
) -> str:
    assistant: AssistAIClient = ctx["assistant"]
    redis = ctx["redis"]
    raw = await redis.get(redis_key)
    try:
        await redis.delete(redis_key)
    except Exception:  # noqa: BLE001, S110
        pass
    if not raw:
        raise ValueError("STT payload missing or expired in Redis")
    audio_bytes = raw if isinstance(raw, (bytes, bytearray)) else bytes(raw)
    logger.info("job_transcribe_audio", bytes=len(audio_bytes), filename=filename)
    return await assistant.transcribe_audio(
        bytes(audio_bytes),
        filename=filename,
        model=model,
        language=language,
    )


class WorkerSettings:
    """Settings consumed by ``arq worker.WorkerSettings``."""

    functions: ClassVar[list] = [
        generate_image_task,
        synthesize_speech_task,
        transcribe_audio_task,
    ]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = redis_settings()
    max_jobs = int(os.getenv("ARQ_MAX_JOBS", "10"))
    job_timeout = int(os.getenv("ARQ_JOB_TIMEOUT", "300"))
