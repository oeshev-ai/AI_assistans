"""Aiogram handlers."""

from __future__ import annotations

from app.bot import runtime as R
from app.bot.constants import (
    _CHROME,
    MAX_IMAGE_SIZE_MB,
    MAX_MEDIA_SIZE_MB,
    AssistAIClient,
    AssistantStorage,
    CallbackQuery,
    FSMContext,
    Message,
    TelegramBadRequest,
    TopicHistoryStorage,
    UploadSessionStore,
    UserMedia,
    VisionError,
    build_image_content,
    build_multimodal_content,
    download_telegram_file,
    ensure_within_size_limit,
    extract_image_from_message,
    io,
    logger,
    process_pdf,
    rate_limiter,
)


async def on_upload_cancel(
    query: CallbackQuery,
    assistant: AssistAIClient,
    state: FSMContext,
) -> None:
    """Отмена облачной загрузки: удаляет сообщения со ссылкой и сессию."""
    await query.answer()
    data = query.data or ""
    upload_id = data.removeprefix("upload:cancel:")
    if not upload_id or query.message is None:
        return
    chat_id = query.message.chat.id
    store = getattr(assistant, "upload_session_store", None)
    bot = query.bot
    if not isinstance(store, UploadSessionStore) or bot is None:
        return
    session = await store.get(upload_id)
    if session is None or session.chat_id != chat_id:
        if isinstance(query.message, Message):
            try:
                await query.message.edit_reply_markup(reply_markup=None)
            except TelegramBadRequest:
                pass
        await bot.send_message(chat_id, "Ссылка уже недействительна или устарела.")
        return
    if session.status not in {"pending"}:
        await bot.send_message(
            chat_id, "Загрузка уже началась или завершена — отменить нельзя."
        )
        return

    await R._delete_chat_messages(bot, chat_id, session.prompt_message_ids)
    await store.delete(upload_id)
    # Убрать id из chrome FSM, если ещё там.
    stored = await state.get_data()
    chrome = stored.get(_CHROME)
    if isinstance(chrome, list) and session.prompt_message_ids:
        leftover = [
            item
            for item in chrome
            if isinstance(item, int) and item not in set(session.prompt_message_ids)
        ]
        await state.update_data({_CHROME: leftover})
    await bot.send_message(chat_id, "Загрузка отменена.")


async def handle_audio(
    message: Message,
    assistant: AssistAIClient,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Аудиофайл → персона (не авто-STT; расшифровка через tool)."""
    audio = message.audio
    if audio is None:
        return
    if await state.get_state() is not None:
        return
    user_id = message.from_user.id if message.from_user else message.chat.id
    if not await rate_limiter.check_text_limit(user_id):
        retry_after = await rate_limiter.get_retry_after(user_id, "text")
        await message.answer(
            f"⏳ Слишком много запросов. Подождите {int(retry_after) or 60} секунд."
        )
        return
    if audio.file_size and audio.file_size > MAX_MEDIA_SIZE_MB * 1024 * 1024:
        await R._offer_cloud_upload_for_oversized(
            message, assistant, state, media_kind="audio"
        )
        return
    bot = message.bot
    if bot is None:
        await message.answer("Не удалось загрузить файл")
        return
    try:
        buffer = io.BytesIO()
        await bot.download(audio, destination=buffer)
        data = buffer.getvalue()
    except Exception:
        logger.exception("audio_download_failed")
        await message.answer("Не удалось скачать аудио.")
        return
    if not data:
        await message.answer("Пустой аудиофайл.")
        return
    filename = audio.file_name or "audio.mp3"
    mime = audio.mime_type or "audio/mpeg"
    await R._process_attached_media(
        message,
        assistant,
        topic_storage,
        assistant_storage,
        state,
        data=data,
        filename=filename,
        mime_type=mime,
        kind="audio",
    )


async def handle_video_file(
    message: Message,
    assistant: AssistAIClient,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Видео / video_note → персона; STT через tool после извлечения аудио."""
    video = message.video or message.video_note
    if video is None:
        return
    if await state.get_state() is not None:
        return
    user_id = message.from_user.id if message.from_user else message.chat.id
    if not await rate_limiter.check_text_limit(user_id):
        retry_after = await rate_limiter.get_retry_after(user_id, "text")
        await message.answer(
            f"⏳ Слишком много запросов. Подождите {int(retry_after) or 60} секунд."
        )
        return
    size = getattr(video, "file_size", None)
    if size and size > MAX_MEDIA_SIZE_MB * 1024 * 1024:
        await R._offer_cloud_upload_for_oversized(
            message, assistant, state, media_kind="video"
        )
        return
    bot = message.bot
    if bot is None:
        await message.answer("Не удалось загрузить файл")
        return
    try:
        buffer = io.BytesIO()
        await bot.download(video, destination=buffer)
        data = buffer.getvalue()
    except Exception:
        logger.exception("video_download_failed")
        await message.answer("Не удалось скачать видео.")
        return
    if not data:
        await message.answer("Пустой видеофайл.")
        return
    filename = getattr(video, "file_name", None) or "video.mp4"
    mime = getattr(video, "mime_type", None) or "video/mp4"
    await R._process_attached_media(
        message,
        assistant,
        topic_storage,
        assistant_storage,
        state,
        data=data,
        filename=filename,
        mime_type=mime,
        kind="video",
    )


async def handle_photo(
    message: Message,
    assistant: AssistAIClient,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Анализирует фото через multimodal stream_chat."""
    if not message.photo:
        return
    if await state.get_state() is not None:
        return
    user_id = message.from_user.id if message.from_user else message.chat.id
    if not await rate_limiter.check_text_limit(user_id):
        retry_after = await rate_limiter.get_retry_after(user_id, "text")
        await message.answer(
            f"⏳ Слишком много запросов. Подождите {int(retry_after) or 60} секунд."
        )
        logger.warning("rate_limit_exceeded", user_id=user_id, operation="text")
        return
    bot = message.bot
    if bot is None:
        await message.answer("Не удалось загрузить файл")
        return
    largest = message.photo[-1]
    if (
        largest.file_size is not None
        and largest.file_size > MAX_IMAGE_SIZE_MB * 1024 * 1024
    ):
        await message.answer(
            f"Файл слишком большой. Максимум — {MAX_IMAGE_SIZE_MB} МБ."
        )
        return
    try:
        data = await download_telegram_file(bot, largest.file_id)
        ensure_within_size_limit(data, label="Фото")
        logger.info(
            "Фото, chat_id=%s size=%s",
            message.chat.id,
            len(data),
        )
        image_part = build_image_content(data, "image/jpeg")
        caption = message.caption or ""
        multimodal = build_multimodal_content(caption, [image_part])
        placeholder = f"[Изображение: {caption.strip() or 'без подписи'}]"
        user_media = [
            UserMedia(
                kind="image",
                filename="photo.jpg",
                mime_type="image/jpeg",
                data=data,
            )
        ]
    except VisionError as exc:
        await message.answer(exc.message)
        return
    await R._process_vision_message(
        message,
        assistant,
        topic_storage,
        assistant_storage,
        state,
        multimodal_content=multimodal,
        history_placeholder=placeholder,
        user_media=user_media,
    )


async def handle_document(
    message: Message,
    assistant: AssistAIClient,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Документ: картинка/PDF через Vision; аудио/видео — в media pipeline персоны."""
    if await state.get_state() is not None:
        return
    user_id = message.from_user.id if message.from_user else message.chat.id
    if not await rate_limiter.check_text_limit(user_id):
        retry_after = await rate_limiter.get_retry_after(user_id, "text")
        await message.answer(
            f"⏳ Слишком много запросов. Подождите {int(retry_after) or 60} секунд."
        )
        logger.warning("rate_limit_exceeded", user_id=user_id, operation="text")
        return

    document = message.document
    bot = message.bot
    if bot is None:
        await message.answer("Не удалось загрузить файл")
        return

    # Видео/аудио, отправленные «как файл», приходят document, а не video/audio.
    if document is not None:
        mime = (document.mime_type or "").strip()
        file_name = document.file_name or "document"
        media_kind = R._document_media_kind(mime, file_name)
        if media_kind is not None:
            if (
                document.file_size is not None
                and document.file_size > MAX_MEDIA_SIZE_MB * 1024 * 1024
            ):
                await R._offer_cloud_upload_for_oversized(
                    message,
                    assistant,
                    state,
                    media_kind="audio" if media_kind == "audio" else "video",
                )
                return
            try:
                data = await download_telegram_file(bot, document.file_id)
            except VisionError as exc:
                await message.answer(exc.message)
                return
            if not data:
                await message.answer("Пустой файл.")
                return
            logger.info(
                "document_media",
                kind=media_kind,
                name=file_name,
                mime=mime or "unknown",
                size=len(data),
            )
            await R._process_attached_media(
                message,
                assistant,
                topic_storage,
                assistant_storage,
                state,
                data=data,
                filename=file_name,
                mime_type=mime
                or ("video/mp4" if media_kind == "video" else "audio/mpeg"),
                kind=media_kind,
            )
            return

    extracted = extract_image_from_message(message)
    if isinstance(extracted, str):
        await message.answer(extracted)
        return
    file_id, mime, file_name = extracted
    if (
        document is not None
        and document.file_size is not None
        and document.file_size > MAX_IMAGE_SIZE_MB * 1024 * 1024
    ):
        await message.answer(
            f"Файл слишком большой. Максимум — {MAX_IMAGE_SIZE_MB} МБ."
        )
        return
    user_media: list[UserMedia] | None = None
    try:
        data = await download_telegram_file(bot, file_id)
        ensure_within_size_limit(data, label="Файл")
        caption = message.caption or ""
        if mime == "application/pdf":
            pages = process_pdf(data)
            logger.info(
                "PDF, chat_id=%s name=%s size=%s pages=%s",
                message.chat.id,
                file_name,
                len(data),
                len(pages),
            )
            image_parts = [build_image_content(page, "image/png") for page in pages]
            multimodal = build_multimodal_content(caption, image_parts)
            placeholder = f"[Документ: {file_name}, {len(pages)} стр.]"
            status_text = "Анализирую документ…"
        else:
            logger.info(
                "Документ-картинка, chat_id=%s name=%s mime=%s size=%s",
                message.chat.id,
                file_name,
                mime,
                len(data),
            )
            image_parts = [build_image_content(data, mime)]
            multimodal = build_multimodal_content(caption, image_parts)
            placeholder = (
                f"[Изображение: {caption.strip() or file_name or 'без подписи'}]"
            )
            status_text = "Анализирую изображение…"
            user_media = [
                UserMedia(
                    kind="image",
                    filename=file_name or "image.bin",
                    mime_type=mime,
                    data=data,
                )
            ]
    except VisionError as exc:
        await message.answer(exc.message)
        return
    await R._process_vision_message(
        message,
        assistant,
        topic_storage,
        assistant_storage,
        state,
        multimodal_content=multimodal,
        history_placeholder=placeholder,
        status_text=status_text,
        user_media=user_media,
    )
