"""Aiogram handlers."""

from __future__ import annotations

from app.bot import runtime as R
from app.bot.constants import (
    _KEEP_OUTGOING,
    MAX_VOICE_DURATION_SECONDS,
    MAX_VOICE_SIZE_MB,
    MSG_INTERNAL,
    SHOW_RECOGNIZED_TEXT,
    VOICE_RECOGNIZED_PREVIEW_LIMIT,
    AssistAIClient,
    AssistAIError,
    AssistantStorage,
    AssistantStorageError,
    BudgetExceededError,
    FSMContext,
    Message,
    Token,
    TopicHistoryStorage,
    cost_tracker,
    httpx,
    job_queue,
    logger,
    rate_limiter,
    security,
    set_billing_user,
)
from app.storage.pending_media import PendingMediaStorage


async def handle_text(
    message: Message,
    assistant: AssistAIClient,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    state: FSMContext,
    pending_media: PendingMediaStorage | None = None,
) -> None:
    """Обычный диалог в активной теме. Ответ языковой модели идёт потоком."""
    user_text = (message.text or "").strip()
    if not user_text:
        return
    user_id = message.from_user.id if message.from_user else message.chat.id
    if not security.validate_message_length(user_text):
        await message.answer("⚠️ Сообщение слишком длинное. Максимум 4000 символов.")
        return
    if pending_media is not None and user_text.casefold() in {
        "отмена",
        "стоп",
        "cancel",
    }:
        from app.bot.photo_flow import consume_pending_text

        if await consume_pending_text(
            message,
            assistant,
            topic_storage,
            assistant_storage,
            state,
            pending_media,
            user_text,
        ):
            return
    sanitized_text = security.sanitize_prompt(user_text)
    from app.bot.album import get_album_hub
    from app.bot.handlers.photo import deliver_album

    # Album flush consumes the text slot inside route_album — avoid a double hit.
    hub = get_album_hub()
    if hub.has_pending(message.chat.id, user_id):
        flushed = await hub.flush_text(
            message.chat.id,
            user_id,
            sanitized_text,
            deliver_album,
        )
        if flushed:
            return

    if not await rate_limiter.check_text_limit(user_id):
        retry_after = await rate_limiter.get_retry_after(user_id, "text")
        await message.answer(
            f"⏳ Слишком много запросов. Подождите {int(retry_after) or 60} секунд."
        )
        logger.warning("rate_limit_exceeded", user_id=user_id, operation="text")
        return
    if pending_media is not None:
        from app.bot.photo_flow import consume_pending_text

        if await consume_pending_text(
            message,
            assistant,
            topic_storage,
            assistant_storage,
            state,
            pending_media,
            sanitized_text,
        ):
            return
    await R._process_user_message(
        message,
        assistant,
        topic_storage,
        assistant_storage,
        state,
        user_text=sanitized_text,
        input_kind="text",
    )


async def handle_voice(
    message: Message,
    assistant: AssistAIClient,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Распознаёт голосовое и передаёт текст в общий pipeline диалога."""
    voice = message.voice
    if voice is None:
        return
    if voice.duration > MAX_VOICE_DURATION_SECONDS:
        await message.answer("Голосовое слишком длинное. Максимум — 10 минут.")
        return
    if (
        voice.file_size is not None
        and voice.file_size > MAX_VOICE_SIZE_MB * 1024 * 1024
    ):
        await message.answer("Голосовое слишком большое. Максимум — 20 МБ.")
        return
    chat_id = message.chat.id
    user_id = message.from_user.id if message.from_user else chat_id
    if not await rate_limiter.check_text_limit(user_id):
        retry_after = await rate_limiter.get_retry_after(user_id, "text")
        await message.answer(
            f"⏳ Слишком много запросов. Подождите {int(retry_after) or 60} секунд."
        )
        logger.warning("rate_limit_exceeded", user_id=user_id, operation="text")
        return
    logger.info(
        "voice_received",
        duration=voice.duration,
        size=voice.file_size,
        user_id=user_id,
        chat_id=chat_id,
    )
    set_billing_user(user_id)
    keep: Token[bool] | None = _KEEP_OUTGOING.set(True)
    status = await message.answer("🎤 Распознаю речь…")
    keep = R._release_outgoing(keep)
    try:
        await cost_tracker.check_budget(
            user_id, cost_tracker.estimate_stt_cost(float(voice.duration))
        )
        audio_bytes, filename = await R._download_voice(message)
        await R._active_profile(chat_id, assistant_storage)
        transcribed = await job_queue.transcribe_audio(
            assistant,
            audio_bytes,
            filename=filename,
            language=None,
        )
        await cost_tracker.record(
            kind="stt",
            amount_usd=cost_tracker.estimate_stt_cost(float(voice.duration)),
            user_id=user_id,
        )
    except BudgetExceededError as exc:
        await R._edit_stream(status, exc.message, "")
        await R._append_chrome(state, [status.message_id])
        return
    except AssistantStorageError:
        await R._edit_stream(status, MSG_INTERNAL, "")
        await R._append_chrome(state, [status.message_id])
        await R._reply_profile_error(message)
        return
    except ValueError as exc:
        await R._edit_stream(status, str(exc) or "Не удалось обработать голосовое.", "")
        await R._append_chrome(state, [status.message_id])
        return
    except AssistAIError as exc:
        logger.error("AssistAI ошибка STT: %s", exc.message)
        await R._edit_stream(
            status, "⚠️ Не удалось распознать речь. Попробуйте ещё раз.", ""
        )
        await R._append_chrome(state, [status.message_id])
        return
    except httpx.TimeoutException:
        await R._edit_stream(status, "⚠️ Превышено время ожидания распознавания.", "")
        await R._append_chrome(state, [status.message_id])
        return
    except Exception:
        logger.exception("Непредвиденная ошибка STT")
        await R._edit_stream(status, MSG_INTERNAL, "")
        await R._append_chrome(state, [status.message_id])
        return
    finally:
        await cost_tracker.release_reservation(user_id)
    logger.info(
        "Распознано, chat_id=%s duration=%s size=%s chars=%s",
        chat_id,
        voice.duration,
        voice.file_size,
        len(transcribed),
    )
    if not transcribed:
        await R._edit_stream(status, "🤷 В голосовом не распознано речи.", "")
        await R._append_chrome(state, [status.message_id])
        return
    sanitized_voice = security.sanitize_prompt(transcribed)
    if SHOW_RECOGNIZED_TEXT:
        preview = sanitized_voice[:VOICE_RECOGNIZED_PREVIEW_LIMIT]
        await R._edit_stream(
            status,
            f"🗣 Вы сказали:\n«{preview}»\n\nГотовлю ответ…",
            "",
        )
    await R._process_user_message(
        message,
        assistant,
        topic_storage,
        assistant_storage,
        state,
        user_text=sanitized_voice,
        input_kind="voice",
        status=status,
    )
