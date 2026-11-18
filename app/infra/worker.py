"""arq worker process for heavy ProxyAPI jobs.

Run (Docker Compose / repo root shim ``worker.py``)::

    arq worker.WorkerSettings

Equivalent package path::

    arq app.infra.worker.WorkerSettings

Requires ``REDIS_URL`` and ``PROXYAPI_API_KEY``. Enable with ``ARQ_ENABLED=1``
and ``docker compose --profile arq up -d``.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urlparse

import structlog
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
load_dotenv()

from app.infra.logging_setup import configure_logging
from app.llm.assist_ai import AssistAIClient, AssistAIConfig

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

    from app.infra.cost_tracker import cost_tracker

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
        except Exception as exc:
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
    role: str | None = None,
) -> dict[str, str]:
    assistant: AssistAIClient = ctx["assistant"]
    logger.info("job_synthesize_speech", chars=len(text), voice=voice, role=role)
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
    }


async def edit_image_task(
    ctx: dict[str, Any],
    redis_key: str,
    prompt: str,
    model: str | None,
    labels: list[str] | None,
    size: str | None = None,
) -> list[str]:
    """Достаёт файлы правки из Redis и вызывает ``edit_images``."""
    assistant: AssistAIClient = ctx["assistant"]
    redis = ctx["redis"]
    raw = await redis.get(redis_key)
    try:
        await redis.delete(redis_key)
    except Exception:
        pass
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
    logger.info("job_edit_image", files=len(files), model=model)
    return await assistant.edit_images(
        files, prompt, model=model, labels=labels, size=size
    )


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
    except Exception:
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
        edit_image_task,
        synthesize_speech_task,
        transcribe_audio_task,
    ]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = redis_settings()
    max_jobs = int(os.getenv("ARQ_MAX_JOBS", "10"))
    job_timeout = int(os.getenv("ARQ_JOB_TIMEOUT", "300"))
