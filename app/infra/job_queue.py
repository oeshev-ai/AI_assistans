"""Optional arq job queue for heavy ProxyAPI operations.

When ``ARQ_ENABLED=1`` and Redis is available, image generation (and optionally
TTS/STT) run in an ``arq`` worker process. Otherwise calls execute in-process
so local development needs no worker.
"""

from __future__ import annotations

import base64
import json
import math
import uuid
from typing import Any
from urllib.parse import unquote, urlparse

import structlog

from app.llm.media_api import SpeechResult, TranscriptionResult

logger = structlog.get_logger("job_queue")

# Создание картинки может занять весь read-timeout, затем опрос — POLZA_TOTAL_WAIT.
_IMAGE_CREATE_TIMEOUT = 180.0


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
    # Пароль в URL percent-encoded: urlparse его не раскодирует.
    password = unquote(parsed.password) if parsed.password else None
    username = unquote(parsed.username) if parsed.username else None
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
    from app.infra.settings import get_settings

    _enabled = get_settings().arq_enabled
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


def queue_pool() -> Any | None:
    """Пул arq, если очередь поднята. На нём же лежат payload и результаты."""
    return _pool


def _billing_user_id(explicit: int | None = None) -> int | None:
    if explicit is not None:
        return explicit
    from app.infra.cost_tracker import get_billing_user

    return get_billing_user()


def _polza_image_wait_needed() -> float:
    """Минимум ожидания: таймаут создания плюс опрос Polza и запас."""
    from app.infra.settings import get_settings

    polza_wait = max(0.0, get_settings().polza_total_wait_seconds)
    return _IMAGE_CREATE_TIMEOUT + polza_wait + 30.0


def validate_arq_image_timeout() -> None:
    """Fail if ``ARQ_JOB_TIMEOUT`` cannot cover Polza poll + margin.

    A shorter worker timeout kills paid image jobs while the bot may still wait
    (or historically gave up early). Call when the arq queue/worker is enabled.
    """
    from app.infra.settings import get_settings

    settings = get_settings()
    arq_timeout = settings.arq_job_timeout_seconds
    needed = _polza_image_wait_needed()
    if arq_timeout > 0 and arq_timeout < needed:
        logger.error(
            "arq_image_timeout_below_polza_wait",
            arq_job_timeout=arq_timeout,
            polza_total_wait=settings.polza_total_wait_seconds,
            needed=needed,
        )
        raise ValueError(
            "ARQ_JOB_TIMEOUT "
            f"({arq_timeout}) must be >= image create + POLZA_TOTAL_WAIT + 30 ({needed})"
        )


def _waves_until_start(queued: int, max_jobs: int) -> int:
    """Волны до старта: занятые слоты и пачки backlog по ``max_jobs``.

    Пустая очередь всё равно одна волна: zcard не видит уже идущие задачи.
    """
    waiting = max(0, queued)
    return 1 + math.ceil(waiting / max(1, max_jobs))


def image_result_timeout(explicit: float | None = None, *, queued: int = 0) -> float:
    """Бот ждёт задачу не меньше, чем воркеру разрешено её выполнять.

    Ниже ``image create + POLZA_TOTAL_WAIT + 30`` не опускается.
    Короткий ``ARQ_JOB_TIMEOUT`` отклоняет ``validate_arq_image_timeout``.
    Если в очереди уже есть задачи, добавляется их ожидание: текущая волна
    и пачки по ``ARQ_MAX_JOBS``, затем своё выполнение. Иначе бот сдаётся
    через один ``job_timeout``, а оплаченная задача остаётся в backlog.
    Явный ``explicit`` не растягивается.
    """
    if explicit is not None:
        return explicit
    from app.infra.settings import get_settings

    settings = get_settings()
    job_timeout = max(0.0, settings.arq_job_timeout_seconds)
    base = max(_polza_image_wait_needed(), job_timeout)
    if queued <= 0:
        return base
    slot = job_timeout if job_timeout > 0 else base
    return base + _waves_until_start(queued, settings.arq_max_jobs) * slot


def audio_result_timeout(explicit: float | None = None) -> float:
    """Ожидание TTS/STT совпадает с таймаутом задачи, чтобы бот не сдался раньше."""
    if explicit is not None:
        return explicit
    from app.infra.settings import get_settings

    return max(120.0, get_settings().arq_job_timeout_seconds)


def _payload_ttl_seconds(queued: int = 0) -> int:
    """TTL байтов edit/STT в Redis, пока job не стартовал.

    Одна волна — уже идущие задачи (``ARQ_JOB_TIMEOUT``). Дальше backlog
    идёт пачками по ``ARQ_MAX_JOBS``. Без этого при всплеске ключ истекает
    раньше старта и воркер получает «payload missing or expired».
    """
    from app.infra.settings import get_settings

    settings = get_settings()
    job_timeout = max(0.0, settings.arq_job_timeout_seconds)
    polza_wait = max(0.0, settings.polza_total_wait_seconds)
    waves = _waves_until_start(queued, settings.arq_max_jobs)
    return int(
        max(
            600.0,
            job_timeout + 120.0,
            polza_wait + 120.0,
            waves * job_timeout + 120.0,
        )
    )


async def _queue_depth(pool: Any) -> int:
    """Сколько задач ждут слот. Уже выполняющиеся в zcard не входят."""
    from arq.constants import default_queue_name

    try:
        depth = await pool.zcard(default_queue_name)
        return max(0, int(depth))
    except Exception as exc:
        logger.warning("arq_queue_depth_failed", error=str(exc))
        return 0


async def _staged_payload_ttl(pool: Any) -> int:
    """TTL с учётом текущего размера очереди arq."""
    return _payload_ttl_seconds(await _queue_depth(pool))


def _result_lost(task: str) -> Exception:
    """Задача уже ушла из очереди, а ключа результата нет."""
    from app.llm.errors import AssistAIError

    logger.error("arq_result_missing", task=task)
    return AssistAIError(
        "Результат не сохранился в очереди. Повтор может списать деньги ещё раз.",
        error_code="QUEUE_RESULT_LOST",
        job_committed=True,
    )


def _job_still_running(task: str) -> Exception:
    """Ключ результата ещё не появился, задача по-прежнему в очереди."""
    from app.llm.errors import AssistAIError

    logger.error("arq_job_still_running", task=task)
    return AssistAIError(
        "Задача ещё выполняется. Повтор может списать деньги ещё раз.",
        error_code="QUEUE_JOB_RUNNING",
        job_committed=True,
    )


async def _job_status(job: Any) -> str:
    status_of = getattr(job, "status", None)
    if status_of is None:
        return ""
    try:
        raw = await status_of()
    except Exception as exc:
        logger.warning("arq_job_status_failed", error=str(exc))
        return ""
    value = getattr(raw, "value", raw)
    return str(value)


async def _await_job_result(job: Any, timeout: float, *, task: str) -> Any:
    """Ждёт результат и не путает пропавший ключ с ещё идущей задачей.

    arq при отсутствии результата и записи в очереди сразу бросает
    ``ResultNotFound``. ``TimeoutError`` бывает только пока задача в очереди.
    """

    from arq.jobs import ResultNotFound

    try:
        return await job.result(timeout=timeout)
    except ResultNotFound as exc:
        raise _result_lost(task) from exc
    except TimeoutError as exc:
        status = await _job_status(job)
        if status == "complete":
            try:
                return await job.result(timeout=0)
            except (TimeoutError, ResultNotFound):
                raise _result_lost(task) from exc
        if status in {"queued", "deferred", "in_progress", ""}:
            raise _job_still_running(task) from exc
        raise _result_lost(task) from exc


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
    background: str | None = None,
) -> list[str]:
    """Generate an image via arq worker or inline client."""
    user_id = _billing_user_id(billing_user_id)
    if is_queue_enabled():
        pool = _pool
        if pool is None:
            raise RuntimeError("Пул arq не поднят")
        wait = image_result_timeout(timeout, queued=await _queue_depth(pool))
        try:
            job = await pool.enqueue_job(
                "generate_image_task",
                prompt,
                model,
                size,
                user_id,
                background,
            )
        except Exception as exc:
            logger.warning("arq_generate_enqueue_failed", error=str(exc))
            raise _arq_enqueue_failed("generate_image_task") from exc
        if job is None:
            raise _arq_enqueue_failed("generate_image_task")
        logger.info("arq_job_enqueued", task="generate_image_task", job_id=job.job_id)
        result = await _await_job_result(job, wait, task="generate_image_task")
        return list(result)
    return await assistant.generate_image(
        prompt=prompt, model=model, size=size, background=background
    )


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
    if is_queue_enabled() and _pool is not None:
        queued = await _queue_depth(_pool)
        wait = image_result_timeout(timeout, queued=queued)
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
            await _pool.set(key, json.dumps(payload), ex=_payload_ttl_seconds(queued))
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
            result = await _await_job_result(job, wait, task="edit_image_task")
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
        pool = _pool
        if pool is None:
            raise RuntimeError("Пул arq не поднят")
        try:
            job = await pool.enqueue_job(
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
        payload = await _await_job_result(job, timeout, task="synthesize_speech_task")
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
            await _pool.set(key, audio_bytes, ex=await _staged_payload_ttl(_pool))
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
            return _coerce_transcription(
                await _await_job_result(job, timeout, task="transcribe_audio_task")
            )
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
