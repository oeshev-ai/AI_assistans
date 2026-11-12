"""Messaging, chrome cleanup, navigation helpers."""

from __future__ import annotations

import app.bot.screens as screens
from app.bot import models as M
from app.bot.constants import (
    _CHROME,
    _CHROME_BUCKET,
    _FLOW_KEYS,
    _KEEP_OUTGOING,
    _MENU_ANCHOR,
    _MENU_ASSISTANTS,
    _MENU_TOPICS,
    _SCREEN_ASSISTANTS,
    _SCREEN_IMAGE,
    _SCREEN_KEYS,
    _SCREEN_PICKER,
    _SCREEN_SETTINGS,
    _SCREEN_TOPICS,
    DEFAULT_ASSISTANT_ID,
    DEFAULT_ASSISTANT_NAME,
    MAX_MEDIA_SIZE_MB,
    MSG_HISTORY,
    MSG_IMAGE_FAILED,
    MSG_INTERNAL,
    MSG_PROFILE,
    MSG_TIMEOUT,
    MSG_UNAVAILABLE,
    STREAM_EDIT_INTERVAL,
    TELEGRAM_TEXT_LIMIT,
    Any,
    AssistAIClient,
    AssistAIError,
    AssistantProfile,
    AssistantStorage,
    AssistantStorageError,
    Awaitable,
    BaseMiddleware,
    Bot,
    BudgetExceededError,
    BufferedInputFile,
    Callable,
    CallbackQuery,
    FSMContext,
    ImageModelScope,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputKind,
    Literal,
    MediaProcessingError,
    MediaTurnScope,
    Message,
    OutboundMedia,
    ReplyKeyboardMarkup,
    S3Storage,
    S3StorageError,
    TelegramBadRequest,
    TelegramObject,
    Token,
    TopicHistoryStorage,
    TopicStorageError,
    UploadMediaKind,
    UploadRequestContext,
    UploadSession,
    UploadSessionStore,
    UserMedia,
    asyncio,
    base64,
    binascii,
    bind_request_context,
    clear_request_context,
    compose_system_prompt,
    cost_tracker,
    httpx,
    image_model_family,
    io,
    is_elevenlabs_tts_model,
    is_speechkit_tts_model,
    is_upload_feature_ready,
    job_queue,
    json,
    logger,
    note_active_user,
    observe_message,
    persona_label,
    prepare_audio_for_stt,
    public_upload_url,
    reset_upload_request_context,
    reset_user_media,
    resolve_speechkit_role,
    sentry_sdk,
    set_billing_user,
    set_upload_request_context,
    set_user_media,
    time,
)
from app.bot.models import (  # noqa: F401
    _chosen_model_line,
    _clip_label,
    _clip_tts_text,
    _compatible_tts_voice,
    _confirm_row,
    _default_image_size,
    _display_image_size,
    _fit_screen,
    _hints_block,
    _image_model_of,
    _menu_button,
    _model_keyboard,
    _optional_model,
    _page_bounds,
    _pager_row,
    _reply_menu,
    _resolve_profile_tts_voice,
    _should_send_voice_reply,
    _size_choices,
    _stored_image_size,
    _text_model_of,
    _topic_display_name,
    _tts_model_of,
    _tts_voices_for,
    _vision_model_of,
)
from app.bot.states import (
    ImageSG,
)


async def _active_profile(chat_id: int, storage: AssistantStorage) -> AssistantProfile:
    """Возвращает активную персону. Пустой чат получает «Стандартный»."""
    profile = await storage.get_active_assistant(chat_id)
    if profile is None:
        raise AssistantStorageError("Активный ассистент не найден.")
    return profile


async def _menu_for(
    chat_id: int,
    assistant_storage: AssistantStorage,
    topic_storage: TopicHistoryStorage,
    *,
    ensure_topic: bool = False,
) -> ReplyKeyboardMarkup:
    """Собирает нижнее меню по активной персоне и её теме."""
    assistant_name = DEFAULT_ASSISTANT_NAME
    assistant_id = DEFAULT_ASSISTANT_ID
    try:
        profile = await _active_profile(chat_id, assistant_storage)
        assistant_name = persona_label(profile.name, profile.icon)
        assistant_id = profile.id
    except AssistantStorageError:
        logger.exception("Меню без профиля, chat_id=%s", chat_id)
    topic_name = "Общая"
    try:
        if ensure_topic:
            topic_id = await topic_storage.ensure_active(chat_id, assistant_id, "Общая")
        else:
            topic_id = await topic_storage.get_active_topic(chat_id, assistant_id)
        topics = await topic_storage.list_topics(chat_id, assistant_id)
        found = next(
            (item["name"] for item in topics if item["id"] == topic_id),
            "",
        )
        if found:
            topic_name = M._topic_display_name(found)
    except TopicStorageError:
        logger.exception("Меню без темы, chat_id=%s", chat_id)
    return M._reply_menu(assistant_name, topic_name)


async def _prepare_messages(
    chat_id: int,
    assistant_id: str,
    user_text: str,
    topic_id: str,
    storage: TopicHistoryStorage,
    profile: AssistantProfile,
) -> list[dict[str, Any]]:
    """Добавляет реплику пользователя и собирает запрос для активной персоны.

    В модель уходит системный промпт профиля и история темы без старых
    ``system``. ``stream_chat`` дописывает tool-сообщения в переданный список,
    поэтому это копия: черновики не попадают в Redis.
    """
    limit = storage.max_length
    await storage.append_message(
        chat_id,
        assistant_id,
        topic_id,
        {"role": "user", "content": user_text},
        limit,
    )
    stored = await storage.get_history(chat_id, assistant_id, topic_id)
    dialog = [dict(item) for item in stored if item.get("role") != "system"]
    if len(dialog) != len(stored):
        # Промпт персоны подставляется только в запрос. Старый system в теме
        # занимает слот trim и расходится с активным ассистентом — убираем.
        await storage.clear_topic(chat_id, assistant_id, topic_id)
        for item in dialog:
            await storage.append_message(chat_id, assistant_id, topic_id, item, limit)
    return [
        {"role": "system", "content": compose_system_prompt(profile.system_prompt)},
        *dialog,
    ]


async def _prepare_messages_with_image(
    chat_id: int,
    assistant_id: str,
    multimodal_content: list[dict[str, Any]],
    history_placeholder: str,
    topic_id: str,
    storage: TopicHistoryStorage,
    profile: AssistantProfile,
) -> list[dict[str, Any]]:
    """Пишет текстовый заменитель в тему, а в модель отдаёт multimodal user."""
    limit = storage.max_length
    await storage.append_message(
        chat_id,
        assistant_id,
        topic_id,
        {"role": "user", "content": history_placeholder},
        limit,
    )
    stored = await storage.get_history(chat_id, assistant_id, topic_id)
    dialog = [dict(item) for item in stored if item.get("role") != "system"]
    if len(dialog) != len(stored):
        await storage.clear_topic(chat_id, assistant_id, topic_id)
        for item in dialog:
            await storage.append_message(chat_id, assistant_id, topic_id, item, limit)
    prior = dialog[:-1] if dialog else []
    return [
        {
            "role": "system",
            "content": compose_system_prompt(profile.system_prompt),
        },
        *prior,
        {"role": "user", "content": multimodal_content},
    ]


def _compact_tool_content(content: str) -> str:
    """Убирает URL и base64 из результата generate_image, оставляя факт успеха."""
    try:
        payload = json.loads(content)
    except ValueError:
        return content
    if not isinstance(payload, dict):
        return content
    raw_images = payload.get("images")
    if not isinstance(raw_images, list):
        return content
    compacted = dict(payload)
    compacted["images"] = [
        {"type": item.get("type")}
        if isinstance(item, dict) and "type" in item
        else {"type": "unknown"}
        for item in raw_images
    ]
    if "status" not in compacted:
        compacted["status"] = "success"
    if "note" not in compacted:
        compacted["note"] = (
            "Изображение уже передано пользователю в Telegram. "
            "Не вызывай generate_image повторно для этого запроса."
        )
    return json.dumps(compacted, ensure_ascii=False)


def _sanitize_history_message(message: dict[str, Any]) -> dict[str, Any] | None:
    """Готовит сообщение раунда модели к записи в тему.

    System и user не сохраняем здесь: user уже записан в ``_prepare_messages``.
    Reasoning убираем, чтобы не тащить его в следующие запросы других моделей.
    """
    role = message.get("role")
    if role not in {"assistant", "tool"}:
        return None
    stored = {
        key: value
        for key, value in message.items()
        if key not in {"reasoning", "reasoning_content"}
    }
    if role == "tool" and isinstance(stored.get("content"), str):
        stored["content"] = _compact_tool_content(stored["content"])
    return stored


def _image_prompts_from_messages(messages: list[dict[str, Any]]) -> list[str]:
    """Достаёт prompt из коротких JSON-ответов generate_image."""
    prompts: list[str] = []
    for item in messages:
        if item.get("role") != "tool":
            continue
        content = item.get("content")
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except ValueError:
            continue
        if not isinstance(payload, dict):
            continue
        prompt = payload.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            prompts.append(prompt.strip())
    return prompts


def _image_completion_text(messages: list[dict[str, Any]], image_count: int) -> str:
    """Текст-заглушка, если модель сгенерировала картинку без финальной реплики."""
    prompts = _image_prompts_from_messages(messages)
    if len(prompts) == 1:
        subject = f"по запросу: «{prompts[0]}»"
    elif prompts:
        joined = "; ".join(prompts[:3])
        subject = f"по запросам: «{joined}»"
    else:
        subject = f"({image_count} шт.)"
    return (
        f"Сгенерировано изображение {subject}. "
        "Картинка уже отправлена пользователю в Telegram. "
        "Повторно вызывать generate_image для этого запроса не нужно."
    )


async def _remember_model_turn(
    chat_id: int,
    assistant_id: str,
    topic_id: str,
    storage: TopicHistoryStorage,
    new_messages: list[dict[str, Any]],
    *,
    answer: str,
    images: list[dict[str, Any]],
) -> None:
    """Пишет в тему tool-раунд и финальный ответ без сырых картинок.

    Без этого при возврате к теме модель снова вызывает generate_image:
    в истории был только user, а успех инструмента терялся вместе с копией
    ``stream_chat``.
    """
    limit = storage.max_length
    to_store: list[dict[str, Any]] = []
    for item in new_messages:
        cleaned = _sanitize_history_message(item)
        if cleaned is not None:
            to_store.append(cleaned)

    final_text = answer.strip()
    if not final_text and images:
        final_text = _image_completion_text(new_messages, len(images))

    final_index: int | None = None
    for index in range(len(to_store) - 1, -1, -1):
        item = to_store[index]
        if item.get("role") == "assistant" and not item.get("tool_calls"):
            final_index = index
            break

    if final_index is not None:
        if final_text:
            to_store[final_index]["content"] = final_text
        else:
            # Пустой финал без картинок в историю не пишем.
            to_store.pop(final_index)
    elif final_text:
        to_store.append({"role": "assistant", "content": final_text})

    if not to_store:
        return

    for item in to_store:
        await storage.append_message(chat_id, assistant_id, topic_id, item, limit)


async def _active_topic(
    chat_id: int, assistant_id: str, storage: TopicHistoryStorage
) -> str:
    """Возвращает активную тему персоны. На первом сообщении создаёт «Общая»."""
    return await storage.ensure_active(chat_id, assistant_id, "Общая")


async def _reply_history_error(message: Message) -> None:
    logger.exception("Ошибка хранилища истории, chat_id=%s", message.chat.id)
    await message.answer(MSG_HISTORY)


async def _reply_profile_error(message: Message) -> None:
    logger.exception("Ошибка хранилища профилей, chat_id=%s", message.chat.id)
    await message.answer(MSG_PROFILE)


def _public_error(exc: BaseException) -> str:
    """Текст для пользователя. Подробности исключения наружу не выходят."""
    cause = exc.__cause__
    if isinstance(exc, httpx.TimeoutException) or isinstance(
        cause, httpx.TimeoutException
    ):
        return MSG_TIMEOUT
    if isinstance(exc, AssistAIError):
        return MSG_UNAVAILABLE
    return MSG_INTERNAL


def _log_handler_error(exc: BaseException) -> None:
    if isinstance(exc, AssistAIError):
        logger.error("AssistAI ошибка: %s", exc.message)
        return
    if isinstance(exc, httpx.TimeoutException):
        logger.error("Превышено время ожидания внешнего сервиса")
        return
    logger.exception("Непредвиденная ошибка")


async def _send_text(message: Message, text: str) -> None:
    """Отправляет текст частями: у Telegram лимит 4096 символов на сообщение."""
    payload = text.strip() or "Пустой ответ."
    offset = 0
    while offset < len(payload):
        await message.answer(payload[offset : offset + TELEGRAM_TEXT_LIMIT])
        offset += TELEGRAM_TEXT_LIMIT


async def _edit_or_answer(
    query: CallbackQuery, text: str, keyboard: InlineKeyboardMarkup | None = None
) -> None:
    """Редактирует сообщение кнопки. Повтор того же текста не считается ошибкой."""
    message = query.message
    if not isinstance(message, Message):
        return
    try:
        await message.edit_text(text, reply_markup=keyboard)
    except TelegramBadRequest as exc:
        if "message is not modified" in str(exc).lower():
            return
        logger.warning("Не удалось отредактировать сообщение: %s", exc)


async def _edit_stream(status: Message, text: str, previous: str) -> str:
    """Обновляет черновик ответа. Повтор того же текста Telegram отвергает."""
    visible = text[:TELEGRAM_TEXT_LIMIT]
    if not visible or visible == previous:
        return previous
    try:
        await status.edit_text(visible)
    except TelegramBadRequest as exc:
        if "message is not modified" in str(exc).lower():
            return previous
        logger.warning("Не удалось обновить сообщение: %s", exc)
        return previous
    return visible


async def _send_overflow(message: Message, text: str) -> None:
    """Доставляет хвост ответа, который не поместился в редактируемое сообщение."""
    rest = text[TELEGRAM_TEXT_LIMIT:]
    while rest:
        chunk = rest[:TELEGRAM_TEXT_LIMIT]
        rest = rest[TELEGRAM_TEXT_LIMIT:]
        await message.answer(chunk)


def _is_http_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


def _extract_generated_images(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Достаёт картинки из результатов инструмента ``generate_image``.

    Имя функции лежит в ``assistant.tool_calls``, а байты или URL — в
    последующем сообщении ``role: tool`` с тем же ``tool_call_id``.
    Битый JSON пропускается: текстовый ответ модели всё равно уйдёт пользователю.
    """
    names_by_id: dict[str, str] = {}
    for item in messages:
        if item.get("role") != "assistant":
            continue
        tool_calls = item.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            call_id = call.get("id")
            function = call.get("function")
            if not isinstance(call_id, str) or not isinstance(function, dict):
                continue
            name = function.get("name")
            if isinstance(name, str):
                names_by_id[call_id] = name

    images: list[dict[str, Any]] = []
    for item in messages:
        if item.get("role") != "tool":
            continue
        call_id = item.get("tool_call_id")
        if not isinstance(call_id, str) or names_by_id.get(call_id) != "generate_image":
            continue
        content = item.get("content")
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except ValueError:
            logger.warning("Не удалось разобрать результат generate_image")
            continue
        if not isinstance(payload, dict):
            continue
        raw_images = payload.get("images")
        if not isinstance(raw_images, list):
            continue
        for image in raw_images:
            if not isinstance(image, dict):
                continue
            kind = image.get("type")
            value = image.get("value")
            if kind in {"url", "base64"} and isinstance(value, str) and value:
                images.append({"type": kind, "value": value})
    return images


async def _send_extracted_images(
    message: Message, images: list[dict[str, Any]]
) -> None:
    """Отправляет картинки из инструмента. Сбой одной не отменяет остальные."""
    for index, image in enumerate(images):
        value = image.get("value")
        if not isinstance(value, str):
            continue
        try:
            await _send_generated_image(message, value, index)
        except Exception:
            logger.exception("Не удалось отправить изображение из generate_image")


async def _send_outbound_media(message: Message, items: list[OutboundMedia]) -> None:
    """Отправляет файлы из media-tools (видео, аудио, транскрипт, правки)."""
    for item in items:
        try:
            if item.kind == "image":
                if item.url:
                    await message.answer_photo(photo=item.url)
                elif item.data:
                    await message.answer_photo(
                        photo=BufferedInputFile(item.data, filename=item.filename)
                    )
            elif item.kind == "audio":
                if not item.data:
                    continue
                audio = BufferedInputFile(item.data, filename=item.filename)
                if item.filename.endswith((".ogg", ".opus")):
                    await message.answer_voice(voice=audio)
                else:
                    await message.answer_audio(audio=audio)
            elif item.kind == "video":
                if not item.data:
                    continue
                await message.answer_video(
                    video=BufferedInputFile(item.data, filename=item.filename)
                )
            elif item.kind == "document":
                if not item.data:
                    continue
                await message.answer_document(
                    document=BufferedInputFile(item.data, filename=item.filename)
                )
        except Exception:
            logger.exception(
                "outbound_media_send_failed", kind=item.kind, filename=item.filename
            )


async def _send_generated_image(message: Message, image: str, index: int) -> None:
    """URL уходит как ссылка, base64 — как файл. Оба варианта отдаёт generate_image."""
    filename = "image.png" if index == 0 else f"image_{index + 1}.png"
    if _is_http_url(image):
        await message.answer_photo(photo=image)
        return
    try:
        raw = base64.b64decode(image, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise AssistAIError("ProxyAPI вернул нечитаемый base64 изображения.") from exc
    if not raw:
        raise AssistAIError("ProxyAPI вернул пустое изображение.")
    await message.answer_photo(photo=BufferedInputFile(raw, filename=filename))


async def _send_voice_reply(
    message: Message,
    assistant: AssistAIClient,
    profile: AssistantProfile,
    text: str,
) -> None:
    """Озвучивает ответ персоны. Сбой TTS не откатывает текст и историю."""
    tts_text = M._clip_tts_text(text)
    if not tts_text:
        return
    model = M._tts_model_of(profile, assistant)
    voice = M._resolve_profile_tts_voice(profile, assistant)
    fmt = profile.tts_format or assistant.config.default_tts_format
    role: str | None = None
    if is_speechkit_tts_model(model):
        fmt = "opus"
        requested_role = profile.tts_role or assistant.config.speechkit_role
        role = resolve_speechkit_role(voice, requested_role) or None
    elif is_elevenlabs_tts_model(model):
        # ProxyAPI ElevenLabs отдаёт MP3; opus/alloy-голоса OpenAI не принимаются.
        fmt = "mp3"
    user_id = message.from_user.id if message.from_user else message.chat.id
    set_billing_user(user_id)
    try:
        await cost_tracker.check_budget(
            user_id, cost_tracker.estimate_tts_cost(tts_text)
        )
        speech = await job_queue.synthesize_speech(
            assistant,
            tts_text,
            model=model,
            voice=voice,
            response_format=fmt,
            role=role,
        )
        await cost_tracker.record(
            kind="tts",
            amount_usd=cost_tracker.estimate_tts_cost(tts_text),
            user_id=user_id,
        )
    except BudgetExceededError as exc:
        await message.answer(exc.message)
        return
    except (AssistAIError, httpx.TimeoutException, ValueError) as exc:
        _log_handler_error(exc)
        await message.answer("⚠️ Не удалось озвучить ответ.")
        return
    except Exception:
        logger.exception("Непредвиденная ошибка TTS")
        await message.answer("⚠️ Не удалось озвучить ответ.")
        return
    audio = BufferedInputFile(speech.audio, filename=speech.filename)
    # MP3 (ElevenLabs) — как audio; OGG — как voice-bubble.
    if speech.mime_type == "audio/mpeg" or speech.filename.casefold().endswith(".mp3"):
        try:
            await message.answer_audio(audio=audio)
        except TelegramBadRequest as exc:
            logger.warning("Не удалось отправить озвучку (audio): %s", exc)
        return
    try:
        await message.answer_voice(voice=audio)
    except TelegramBadRequest as exc:
        logger.warning("Telegram не принял voice, пробую audio: %s", exc)
        try:
            await message.answer_audio(
                audio=BufferedInputFile(speech.audio, filename=speech.filename)
            )
        except TelegramBadRequest as nested:
            logger.warning("Не удалось отправить озвучку: %s", nested)


async def _process_user_message(
    message: Message,
    assistant: AssistAIClient,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    state: FSMContext,
    user_text: str,
    input_kind: InputKind = "text",
    status: Message | None = None,
    user_media: list[UserMedia] | None = None,
) -> None:
    """Общий pipeline текста и распознанного голоса в активной теме."""
    chat_id = message.chat.id
    user_id = message.from_user.id if message.from_user else chat_id
    started = time.perf_counter()
    bind_request_context(user_id=user_id, chat_id=chat_id)
    set_billing_user(user_id)
    note_active_user(user_id)
    sentry_sdk.set_user({"id": str(user_id)})
    sentry_sdk.set_context("chat", {"id": chat_id, "type": message.chat.type})
    media_token = set_user_media(user_media or [])
    upload_ctx_token = set_upload_request_context(
        UploadRequestContext(chat_id=chat_id, user_id=user_id)
    )
    try:
        await cost_tracker.check_budget(
            user_id,
            cost_tracker.estimate_chat_cost(
                input_chars=len(user_text), output_chars=500
            ),
        )
    except BudgetExceededError as exc:
        await message.answer(exc.message)
        reset_user_media(media_token)
        reset_upload_request_context(upload_ctx_token)
        clear_request_context()
        return
    try:
        profile = await _active_profile(chat_id, assistant_storage)
        topic_id = await _active_topic(chat_id, profile.id, topic_storage)
        bind_request_context(
            user_id=user_id,
            chat_id=chat_id,
            assistant_id=profile.id,
            topic_id=topic_id,
        )
        history_copy = await _prepare_messages(
            chat_id, profile.id, user_text, topic_id, topic_storage, profile
        )
    except AssistantStorageError:
        reset_upload_request_context(upload_ctx_token)
        await _reply_profile_error(message)
        return
    except TopicStorageError:
        reset_upload_request_context(upload_ctx_token)
        await _reply_history_error(message)
        return
    text_model = M._text_model_of(profile, assistant, chat_id=chat_id)
    logger.info(
        "processing_message",
        kind=input_kind,
        model=text_model,
        message_count=len(history_copy),
    )
    keep: Token[bool] | None = _KEEP_OUTGOING.set(True)
    if status is None:
        status = await message.answer("Готовлю ответ…")
    parts: list[str] = []
    shown = ""
    last_edit = 0.0
    scope = ImageModelScope(profile.image_model, profile.image_size)
    media_scope = MediaTurnScope()
    turn_start = len(history_copy)
    try:
        if message.bot is not None:
            await message.bot.send_chat_action(chat_id=chat_id, action="typing")
        with scope, media_scope:
            async for delta in assistant.stream_chat(
                messages=history_copy, model=text_model
            ):
                if delta.kind != "content" or not delta.text:
                    continue
                parts.append(delta.text)
                now = asyncio.get_running_loop().time()
                if now - last_edit < STREAM_EDIT_INTERVAL:
                    continue
                shown = await _edit_stream(status, "".join(parts), shown)
                last_edit = now
        answer = "".join(parts).strip()
        images = scope.images
        outbound = media_scope.bucket.items
        if not answer and not images and not outbound:
            await _rollback_user(chat_id, profile.id, topic_storage, topic_id, message)
            await _edit_stream(status, "Пустой ответ.", shown)
            keep = _release_outgoing(keep)
            await _append_chrome(state, [status.message_id])
            return
        if answer:
            visible = await _edit_stream(status, answer, shown)
            head = answer[:TELEGRAM_TEXT_LIMIT]
            if visible == head:
                await _send_overflow(message, answer)
            else:
                await _send_text(message, answer)
        else:
            await _delete_bot_message(status)
        keep = _release_outgoing(keep)
        try:
            await _remember_model_turn(
                chat_id,
                profile.id,
                topic_id,
                topic_storage,
                history_copy[turn_start:],
                answer=answer,
                images=images,
            )
        except TopicStorageError:
            fallback = answer.strip()
            if not fallback and images:
                fallback = _image_completion_text(
                    history_copy[turn_start:], len(images)
                )
            if fallback:
                try:
                    await topic_storage.append_message(
                        chat_id,
                        profile.id,
                        topic_id,
                        {"role": "assistant", "content": fallback},
                        topic_storage.max_length,
                    )
                except TopicStorageError:
                    await _reply_history_error(message)
            else:
                await _reply_history_error(message)
        await _send_extracted_images(message, images)
        await _send_outbound_media(message, outbound)
        logger.info(
            "message_answered",
            chars=len(answer),
            images=len(images),
            outbound=len(outbound),
        )
        if answer and M._should_send_voice_reply(profile, input_kind):
            await _send_voice_reply(message, assistant, profile, answer)
    except (AssistAIError, httpx.TimeoutException) as exc:
        keep = _release_outgoing(keep)
        await _send_extracted_images(message, scope.images)
        await _send_outbound_media(message, media_scope.bucket.items)
        _log_handler_error(exc)
        sentry_sdk.capture_exception(exc)
        await _rollback_user(chat_id, profile.id, topic_storage, topic_id, message)
        await _report_stream_error(message, status, parts, _public_error(exc), state)
    except Exception as exc:
        keep = _release_outgoing(keep)
        await _send_extracted_images(message, scope.images)
        await _send_outbound_media(message, media_scope.bucket.items)
        _log_handler_error(exc)
        sentry_sdk.capture_exception(exc)
        await _rollback_user(chat_id, profile.id, topic_storage, topic_id, message)
        await _report_stream_error(message, status, parts, MSG_INTERNAL, state)
    finally:
        reset_user_media(media_token)
        reset_upload_request_context(upload_ctx_token)
        observe_message(input_kind, time.perf_counter() - started)
        clear_request_context()
        _release_outgoing(keep)


async def _download_voice(message: Message) -> tuple[bytes, str]:
    """Скачивает голосовое сообщение Telegram в память.

    Returns:
        ``(audio_bytes, filename)``.

    Raises:
        ValueError: нет voice или пустые байты.
        AssistAIError: сбой скачивания.
    """
    voice = message.voice
    if voice is None:
        raise ValueError("В сообщении нет голосового вложения.")
    bot = message.bot
    if bot is None:
        raise AssistAIError("Бот недоступен для скачивания голосового.")
    buffer = io.BytesIO()
    try:
        await bot.download(voice, destination=buffer)
    except Exception as exc:
        logger.exception("Не удалось скачать голосовое")
        raise AssistAIError("Не удалось скачать голосовое сообщение.") from exc
    audio_bytes = buffer.getvalue()
    if not audio_bytes:
        raise ValueError("Голосовое сообщение пустое.")
    return audio_bytes, "voice.ogg"


async def _delete_chat_messages(
    bot: Bot, chat_id: int, message_ids: list[int] | None
) -> None:
    """Best-effort удаление служебных сообщений (ссылка на загрузку и т.п.)."""
    if not message_ids:
        return
    for message_id in message_ids:
        try:
            await bot.delete_message(chat_id=chat_id, message_id=message_id)
        except TelegramBadRequest:
            continue
        except Exception:
            logger.debug(
                "upload_prompt_delete_failed", chat_id=chat_id, message_id=message_id
            )


async def _offer_cloud_upload_for_oversized(
    message: Message,
    assistant: AssistAIClient,
    state: FSMContext,
    *,
    media_kind: UploadMediaKind,
) -> None:
    """После отказа Telegram getFile — пояснение + ссылка и кнопка «Отмена»."""
    notice = await message.answer(
        f"Файл слишком большой для Telegram (лимит ~{MAX_MEDIA_SIZE_MB} МБ).\n"
        "Его нельзя скачать через обычное вложение."
    )
    chrome_ids = [notice.message_id]

    if not is_upload_feature_ready():
        tip = await message.answer(
            "Облачная загрузка сейчас не настроена.\n"
            f"Сожмите файл до {MAX_MEDIA_SIZE_MB} МБ или напишите администратору."
        )
        chrome_ids.append(tip.message_id)
        await _append_chrome(state, chrome_ids)
        return

    store = getattr(assistant, "upload_session_store", None)
    if not isinstance(store, UploadSessionStore):
        tip = await message.answer(
            "Сервис загрузки временно недоступен. Попробуйте позже."
        )
        chrome_ids.append(tip.message_id)
        await _append_chrome(state, chrome_ids)
        return

    user_id = message.from_user.id if message.from_user else message.chat.id
    session = await store.create(
        chat_id=message.chat.id,
        user_id=user_id,
        purpose="transcribe",
        media_kind=media_kind,
    )
    url = public_upload_url(session.upload_id, session.token)
    if not url:
        tip = await message.answer(
            "Не удалось сформировать ссылку. Напишите администратору."
        )
        chrome_ids.append(tip.message_id)
        await store.delete(session.upload_id)
        await _append_chrome(state, chrome_ids)
        return

    link_msg = await message.answer(
        "Загрузите файл по ссылке (действует около часа):\n"
        f"{url}\n\n"
        "Откройте ссылку → выберите файл → дождитесь «Готово». "
        "Расшифровка придёт в этот чат.",
        reply_markup=InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="Отмена",
                        callback_data=f"upload:cancel:{session.upload_id}",
                    )
                ]
            ]
        ),
        disable_web_page_preview=True,
    )
    chrome_ids.append(link_msg.message_id)
    session.prompt_message_ids = list(chrome_ids)
    await store.update(session)
    await _append_chrome(state, chrome_ids)


def _infer_upload_media_kind(*, mime_type: str, filename: str, preferred: str) -> str:
    if preferred in {"audio", "video"}:
        return preferred
    mime = (mime_type or "").lower()
    name = (filename or "").lower()
    if mime.startswith("video/") or name.endswith((".mp4", ".mov", ".mkv")):
        return "video"
    if name.endswith(".webm") and not mime.startswith("audio/"):
        return "video"
    return "audio"


async def _process_s3_upload_session(
    *,
    bot: Bot,
    assistant: AssistAIClient,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    store: UploadSessionStore,
    storage: S3Storage,
    session: UploadSession,
) -> None:
    """Скачивает объект из бакета и делает STT → transcript.txt в чат."""
    chat_id = session.chat_id
    object_key = session.object_key
    filename = session.filename or "media.bin"
    mime_type = session.mime_type or "application/octet-stream"
    if not object_key:
        await bot.send_message(
            chat_id, "Загрузка без object key — нечего обрабатывать."
        )
        session.status = "failed"
        session.error = "missing_object_key"
        await store.update(session)
        return

    status_msg = await bot.send_message(
        chat_id, f"Файл «{filename}» получен. Расшифровываю…"
    )
    data: bytes | None = None
    try:
        data = await storage.download_bytes(object_key)
        kind = _infer_upload_media_kind(
            mime_type=mime_type,
            filename=filename,
            preferred=session.media_kind,
        )
        chunks = await prepare_audio_for_stt(
            data,
            filename=filename,
            is_video=kind == "video",
        )
        parts: list[str] = []
        for audio_bytes, chunk_name in chunks:
            text = await job_queue.transcribe_audio(
                assistant,
                audio_bytes,
                filename=chunk_name,
                language=None,
            )
            if text:
                parts.append(text)
        transcript = "\n\n".join(parts).strip()
        await _delete_chat_messages(bot, chat_id, session.prompt_message_ids)
        session.prompt_message_ids = None
        await store.update(session)

        if not transcript:
            await bot.send_message(chat_id, "Речь не распознана.")
            session.status = "done"
            await store.update(session)
            return

        await bot.send_document(
            chat_id,
            document=BufferedInputFile(
                transcript.encode("utf-8"), filename="transcript.txt"
            ),
            caption=f"Расшифровка «{filename}» ({len(transcript)} симв.)",
        )
        try:
            await status_msg.delete()
        except Exception:
            logger.debug("upload_status_delete_failed", chat_id=chat_id)

        # Короткая запись в историю активной темы.
        try:
            profile = await _active_profile(chat_id, assistant_storage)
            topic_id = await _active_topic(chat_id, profile.id, topic_storage)
            await topic_storage.append_message(
                chat_id,
                profile.id,
                topic_id,
                {
                    "role": "user",
                    "content": f"[Большой файл загружен в облако: {filename}]",
                },
                topic_storage.max_length,
            )
            await topic_storage.append_message(
                chat_id,
                profile.id,
                topic_id,
                {
                    "role": "assistant",
                    "content": (
                        f"Расшифровка «{filename}» отправлена файлом transcript.txt "
                        f"({len(transcript)} символов)."
                    ),
                },
                topic_storage.max_length,
            )
        except Exception:
            logger.exception("upload_history_append_failed", chat_id=chat_id)

        session.status = "done"
        await store.update(session)
        logger.info(
            "s3_upload_transcribed",
            upload_id=session.upload_id,
            chat_id=chat_id,
            chars=len(transcript),
            chunks=len(chunks),
        )
    except (S3StorageError, MediaProcessingError, AssistAIError) as exc:
        message = getattr(exc, "message", None) or str(exc)
        logger.exception("s3_upload_process_failed", upload_id=session.upload_id)
        await bot.send_message(chat_id, f"Не удалось обработать файл: {message}")
        session.status = "failed"
        session.error = message[:500]
        await store.update(session)
    except Exception as exc:
        logger.exception("s3_upload_process_failed", upload_id=session.upload_id)
        sentry_sdk.capture_exception(exc)
        await bot.send_message(chat_id, "Не удалось обработать загруженный файл.")
        session.status = "failed"
        session.error = "internal"
        await store.update(session)
    finally:
        data = None
        try:
            await storage.delete_object(object_key)
        except Exception:
            logger.warning("s3_cleanup_failed", key=object_key)


async def _process_attached_media(
    message: Message,
    assistant: AssistAIClient,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    state: FSMContext,
    *,
    data: bytes,
    filename: str,
    mime_type: str,
    kind: str,
) -> None:
    """Передаёт аудио/видео персоне: tools сами решают (STT и т.д.)."""
    caption = (message.caption or "").strip()
    if kind == "audio":
        placeholder = f"[Аудиофайл: {filename}]"
        default_ask = (
            "Пользователь прислал аудиофайл. Если нужна расшифровка — "
            "вызови transcribe_media."
        )
        media = [
            UserMedia(kind="audio", filename=filename, mime_type=mime_type, data=data)
        ]
    else:
        placeholder = f"[Видеофайл: {filename}]"
        default_ask = (
            "Пользователь прислал видео. Если нужна расшифровка речи — "
            "вызови transcribe_media (аудио извлекается автоматически)."
        )
        media = [
            UserMedia(kind="video", filename=filename, mime_type=mime_type, data=data)
        ]
    user_text = caption if caption else f"{placeholder}\n{default_ask}"
    status = await message.answer("Обрабатываю файл…")
    await _process_user_message(
        message,
        assistant,
        topic_storage,
        assistant_storage,
        state,
        user_text=user_text,
        input_kind="text",
        status=status,
        user_media=media,
    )


async def _process_vision_message(
    message: Message,
    assistant: AssistAIClient,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    state: FSMContext,
    *,
    multimodal_content: list[dict[str, Any]],
    history_placeholder: str,
    status_text: str = "Анализирую изображение…",
    user_media: list[UserMedia] | None = None,
) -> None:
    """Стримит ответ модели по multimodal-запросу; в тему пишет только текст."""
    chat_id = message.chat.id
    media_token = set_user_media(user_media or [])
    try:
        profile = await _active_profile(chat_id, assistant_storage)
        topic_id = await _active_topic(chat_id, profile.id, topic_storage)
        history_copy = await _prepare_messages_with_image(
            chat_id,
            profile.id,
            multimodal_content,
            history_placeholder,
            topic_id,
            topic_storage,
            profile,
        )
    except AssistantStorageError:
        reset_user_media(media_token)
        await _reply_profile_error(message)
        return
    except TopicStorageError:
        reset_user_media(media_token)
        await _reply_history_error(message)
        return
    vision_model = M._vision_model_of(profile, assistant)
    logger.info(
        "Vision, chat_id=%s topic_id=%s, модель=%s, сообщений в запросе=%s",
        chat_id,
        topic_id,
        vision_model,
        len(history_copy),
    )
    keep: Token[bool] | None = _KEEP_OUTGOING.set(True)
    status = await message.answer(status_text)
    parts: list[str] = []
    shown = ""
    last_edit = 0.0
    scope = ImageModelScope(profile.image_model, profile.image_size)
    media_scope = MediaTurnScope()
    turn_start = len(history_copy)
    try:
        if message.bot is not None:
            await message.bot.send_chat_action(chat_id=chat_id, action="typing")
        with scope, media_scope:
            async for delta in assistant.stream_chat(
                messages=history_copy, model=vision_model
            ):
                if delta.kind != "content" or not delta.text:
                    continue
                parts.append(delta.text)
                now = asyncio.get_running_loop().time()
                if now - last_edit < STREAM_EDIT_INTERVAL:
                    continue
                shown = await _edit_stream(status, "".join(parts), shown)
                last_edit = now
        answer = "".join(parts).strip()
        images = scope.images
        outbound = media_scope.bucket.items
        if not answer and not images and not outbound:
            await _rollback_user(chat_id, profile.id, topic_storage, topic_id, message)
            await _edit_stream(status, "Пустой ответ.", shown)
            keep = _release_outgoing(keep)
            await _append_chrome(state, [status.message_id])
            return
        if answer:
            visible = await _edit_stream(status, answer, shown)
            head = answer[:TELEGRAM_TEXT_LIMIT]
            if visible == head:
                await _send_overflow(message, answer)
            else:
                await _send_text(message, answer)
        else:
            await _delete_bot_message(status)
        keep = _release_outgoing(keep)
        try:
            await _remember_model_turn(
                chat_id,
                profile.id,
                topic_id,
                topic_storage,
                history_copy[turn_start:],
                answer=answer,
                images=images,
            )
        except TopicStorageError:
            fallback = answer.strip()
            if not fallback and images:
                fallback = _image_completion_text(
                    history_copy[turn_start:], len(images)
                )
            if fallback:
                try:
                    await topic_storage.append_message(
                        chat_id,
                        profile.id,
                        topic_id,
                        {"role": "assistant", "content": fallback},
                        topic_storage.max_length,
                    )
                except TopicStorageError:
                    await _reply_history_error(message)
            else:
                await _reply_history_error(message)
        await _send_extracted_images(message, images)
        await _send_outbound_media(message, outbound)
        logger.info(
            "Vision-ответ, chat_id=%s topic_id=%s, символов=%s, картинок=%s",
            chat_id,
            topic_id,
            len(answer),
            len(images),
        )
        if answer and M._should_send_voice_reply(profile, "text"):
            await _send_voice_reply(message, assistant, profile, answer)
    except (AssistAIError, httpx.TimeoutException) as exc:
        keep = _release_outgoing(keep)
        await _send_extracted_images(message, scope.images)
        await _send_outbound_media(message, media_scope.bucket.items)
        _log_handler_error(exc)
        await _rollback_user(chat_id, profile.id, topic_storage, topic_id, message)
        await _report_stream_error(message, status, parts, _public_error(exc), state)
    except Exception as exc:
        keep = _release_outgoing(keep)
        await _send_extracted_images(message, scope.images)
        await _send_outbound_media(message, media_scope.bucket.items)
        _log_handler_error(exc)
        await _rollback_user(chat_id, profile.id, topic_storage, topic_id, message)
        await _report_stream_error(message, status, parts, MSG_UNAVAILABLE, state)
    finally:
        reset_user_media(media_token)
        _release_outgoing(keep)


_DocumentMediaKind = Literal["audio", "video"]


def _document_media_kind(mime: str, file_name: str) -> _DocumentMediaKind | None:
    """Определяет аудио/видео, присланные как document (файл без сжатия)."""
    mime_l = mime.strip().casefold()
    name_l = file_name.strip().casefold()
    video_ext = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".3gp")
    audio_ext = (".mp3", ".wav", ".m4a", ".ogg", ".oga", ".flac", ".aac", ".opus")
    if mime_l.startswith("video/") or name_l.endswith(video_ext):
        return "video"
    if mime_l.startswith("audio/") or name_l.endswith(audio_ext):
        return "audio"
    return None


async def _rollback_user(
    chat_id: int,
    assistant_id: str,
    storage: TopicHistoryStorage,
    topic_id: str,
    message: Message,
) -> None:
    try:
        await storage.rollback_last_user(chat_id, assistant_id, topic_id)
    except TopicStorageError:
        await _reply_history_error(message)


async def _report_stream_error(
    message: Message,
    status: Message,
    parts: list[str],
    text: str,
    state: FSMContext,
) -> None:
    if parts:
        await _delete_bot_message(status)
        await message.answer(text)
        return
    edited = await _edit_stream(status, text, "")
    if edited:
        await _append_chrome(state, [status.message_id])
        return
    await _delete_bot_message(status)
    await message.answer(text)


async def _apply_model_choice(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    state: FSMContext,
    *,
    catalog: tuple[str, ...],
    prefix: str,
    kind: str,
) -> None:
    message = query.message
    if not isinstance(message, Message):
        await query.answer()
        return
    raw = (query.data or "").removeprefix(prefix)
    try:
        selected: str | None = catalog[int(raw)]
    except (ValueError, IndexError):
        await query.answer("Неизвестная модель", show_alert=True)
        return
    chat_id = message.chat.id
    try:
        profile = await _active_profile(chat_id, assistant_storage)
        fields: dict[str, Any]
        if kind == "image":
            old_model = M._image_model_of(profile, assistant, chat_id=chat_id)
            new_model = selected or assistant.config.default_image_model
            fields = {"image_model": selected}
            if image_model_family(old_model) != image_model_family(new_model):
                fields["image_size"] = M._default_image_size(new_model)
            updated = profile.model_copy(update=fields)
        elif kind == "tts":
            effective = selected or assistant.config.default_tts_model
            new_voice = M._compatible_tts_voice(profile, assistant, new_model=selected)
            fields = {
                "tts_model": selected,
                "tts_voice": new_voice,
            }
            if is_speechkit_tts_model(effective):
                fields["tts_format"] = "opus"
                voice_for_role = new_voice or (
                    assistant.config.default_tts_voice
                    if is_speechkit_tts_model(assistant.config.default_tts_model)
                    else "marina"
                )
                fields["tts_role"] = (
                    resolve_speechkit_role(voice_for_role, profile.tts_role) or None
                )
            else:
                fields["tts_role"] = None
                if is_elevenlabs_tts_model(effective):
                    fields["tts_format"] = "mp3"
            updated = profile.model_copy(update=fields)
        else:
            updated = profile.model_copy(update={"text_model": selected})
        await assistant_storage.save_assistant(chat_id, updated)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    await query.answer("Модель обновлена")
    if kind in {"image", "tts"}:
        text, markup = screens._model_picker_parts(updated, assistant, kind)
        await _edit_inline(message, text, markup)
        return
    await _return_to_settings(message, state, updated, assistant)


async def _return_to_settings(
    message: Message,
    state: FSMContext,
    profile: AssistantProfile,
    assistant: AssistAIClient,
) -> None:
    """Убирает выбор модели и обновляет экран настройки персоны."""
    text, markup = screens._settings_persona_view(profile, assistant)
    try:
        await message.delete()
    except TelegramBadRequest as exc:
        logger.warning("Не удалось удалить выбор модели: %s", exc)
    data = await state.get_data()
    if _SCREEN_PICKER in data:
        data.pop(_SCREEN_PICKER, None)
        await state.set_data(data)
    settings_id = data.get(_SCREEN_SETTINGS)
    bot = message.bot
    if (
        isinstance(settings_id, int)
        and bot is not None
        and settings_id != message.message_id
    ):
        try:
            await bot.edit_message_text(
                text,
                chat_id=message.chat.id,
                message_id=settings_id,
                reply_markup=markup,
            )
            return
        except TelegramBadRequest as exc:
            lowered = str(exc).lower()
            if "message is not modified" in lowered:
                return
            logger.warning("Не удалось вернуть настройки: %s", exc)
    sent = await message.answer(text, reply_markup=markup)
    await state.update_data({_SCREEN_SETTINGS: sent.message_id})


def _query_message(query: CallbackQuery) -> Message | None:
    """Сообщение кнопки. Служебные callback без сообщения пропускаются."""
    message = query.message
    if isinstance(message, Message):
        return message
    return None


def _release_outgoing(token: Token[bool] | None) -> Token[bool] | None:
    """Снимает запрет учёта исходящих сообщений, если он ещё включён."""
    if token is not None:
        _KEEP_OUTGOING.reset(token)
    return None


def _is_menu_button(text: str) -> bool:
    """Нижние кнопки «Персоны» / «Темы». Это не вопрос к персоне."""
    return text.startswith(
        (
            f"{_MENU_ASSISTANTS} [",
            f"{_MENU_TOPICS} [",
            # Старые клавиатуры: «Ассистенты» и круглые скобки.
            f"{_MENU_ASSISTANTS} (",
            f"{_MENU_TOPICS} (",
            "Ассистенты [",
            "Ассистенты (",
        )
    )


async def _append_chrome(state: FSMContext, ids: list[int]) -> None:
    """Запоминает служебные сообщения бота, чтобы позже убрать их из чата."""
    if not ids:
        return
    data = await state.get_data()
    current = data.get(_CHROME)
    merged: list[int] = []
    if isinstance(current, list):
        merged.extend(item for item in current if isinstance(item, int))
    for item in ids:
        if item not in merged:
            merged.append(item)
    if len(merged) > 50:
        merged = merged[-50:]
    await state.update_data({_CHROME: merged})


async def _reset_flow(state: FSMContext) -> None:
    """Сбрасывает шаг мастера и сохраняет id сообщения, которое держит меню."""
    data = await state.get_data()
    anchor = data.get(_MENU_ANCHOR)
    chrome = data.get(_CHROME)
    await state.clear()
    kept: dict[str, Any] = {}
    if isinstance(anchor, int):
        kept[_MENU_ANCHOR] = anchor
    if isinstance(chrome, list):
        kept[_CHROME] = chrome
    if kept:
        await state.update_data(kept)


async def _delete_bot_message(message: Message) -> None:
    """Удаляет сообщение бота. Повторное удаление не считается сбоем."""
    try:
        await message.delete()
    except TelegramBadRequest as exc:
        logger.warning("Не удалось удалить сообщение: %s", exc)


async def _apply_reply_menu(
    message: Message, menu: ReplyKeyboardMarkup, state: FSMContext
) -> None:
    """Ставит нижние кнопки.

    Сообщение с клавиатурой нельзя удалять: вместе с ним клиент убирает меню.
    Текст невидимый, чтобы в чате не было отдельной реплики. Прошлое такое
    сообщение удаляется, когда новое уже держит меню.
    """
    token = _KEEP_OUTGOING.set(True)
    try:
        sent = await message.answer("\u2060", reply_markup=menu)
    finally:
        _KEEP_OUTGOING.reset(token)
    data = await state.get_data()
    previous = data.get(_MENU_ANCHOR)
    await state.update_data({_MENU_ANCHOR: sent.message_id})
    if (
        isinstance(previous, int)
        and previous != sent.message_id
        and message.bot is not None
    ):
        try:
            await message.bot.delete_message(
                chat_id=message.chat.id, message_id=previous
            )
        except TelegramBadRequest as exc:
            logger.warning("Не удалось убрать прежнее сообщение меню: %s", exc)


async def _sweep_chrome(
    message: Message, state: FSMContext, data: dict[str, Any]
) -> None:
    """Удаляет меню, настройки и прочие служебные реплики.

    Перед удалением заново ставит нижние кнопки: иначе клиент прячет меню
    вместе с последним сообщением, на котором оно висело.
    """
    stored = await state.get_data()
    raw_ids = stored.get(_CHROME)
    if not isinstance(raw_ids, list) or not raw_ids:
        return
    profiles = data.get("assistant_storage")
    topics = data.get("topic_storage")
    if isinstance(profiles, AssistantStorage) and isinstance(
        topics, TopicHistoryStorage
    ):
        menu = await _menu_for(message.chat.id, profiles, topics)
        await _apply_reply_menu(message, menu, state)
    stored = await state.get_data()
    anchor = stored.get(_MENU_ANCHOR)
    bot = message.bot
    if bot is not None:
        for item in raw_ids:
            if not isinstance(item, int) or item == anchor:
                continue
            try:
                await bot.delete_message(chat_id=message.chat.id, message_id=item)
            except TelegramBadRequest:
                logger.debug(
                    "Служебное сообщение уже удалено chat_id=%s message_id=%s",
                    message.chat.id,
                    item,
                )
    refreshed = await state.get_data()
    for key in _SCREEN_KEYS:
        refreshed.pop(key, None)
    refreshed[_CHROME] = []
    await state.set_data(refreshed)


async def _cleanup_wizard_ui(
    message: Message,
    state: FSMContext,
    assistant_storage: AssistantStorage,
    topic_storage: TopicHistoryStorage,
    *,
    also_delete: list[int] | None = None,
) -> None:
    """Сразу убирает UI мастера. Нужно после шагов, которые заканчиваются callback.

    Обычная очистка срабатывает только на Message; «Готово» при создании
    персоны — callback, поэтому экран моделей иначе остаётся в чате.
    """
    stored = await state.get_data()
    merged: list[int] = []
    raw_ids = stored.get(_CHROME)
    if isinstance(raw_ids, list):
        merged.extend(item for item in raw_ids if isinstance(item, int))
    for item in also_delete or []:
        if item not in merged:
            merged.append(item)
    await state.update_data({_CHROME: merged})
    await _sweep_chrome(
        message,
        state,
        {
            "assistant_storage": assistant_storage,
            "topic_storage": topic_storage,
        },
    )


async def _drop_user_noise(message: Message, state: FSMContext) -> None:
    """Убирает команды, нажатия меню и шаги мастера. Вопрос к персоне остаётся."""
    text = message.text or ""
    if not text:
        return
    current = await state.get_state()
    if current is None and not text.startswith("/") and not _is_menu_button(text):
        return
    try:
        await message.delete()
    except TelegramBadRequest as exc:
        logger.warning("Не удалось убрать служебную реплику пользователя: %s", exc)


class _CleanChat(BaseMiddleware):
    """Перед новым сообщением убирает из чата всё, кроме диалога и картинок."""

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        state = data.get("state")
        bucket: list[int] = []
        token = _CHROME_BUCKET.set(bucket)
        try:
            if isinstance(event, Message) and isinstance(state, FSMContext):
                current = await state.get_state()
                if current is not None and (event.photo or event.document):
                    keep = _KEEP_OUTGOING.set(True)
                    try:
                        await event.answer(
                            "Сначала завершите текущий шаг или нажмите /cancel."
                        )
                    finally:
                        _release_outgoing(keep)
                    return None
                await _sweep_chrome(event, state, data)
                await _drop_user_noise(event, state)
            return await handler(event, data)
        finally:
            if isinstance(state, FSMContext):
                await _append_chrome(state, bucket)
            _CHROME_BUCKET.reset(token)


async def _forget_message_screens(
    state: FSMContext, message_id: int, *extra_keys: str
) -> None:
    """Сбрасывает screen_* (и доп. ключи), если они указывают на это сообщение."""
    data = await state.get_data()
    changed = False
    for key in (*_SCREEN_KEYS, *extra_keys):
        if data.get(key) == message_id:
            data.pop(key, None)
            changed = True
    if changed:
        await state.set_data(data)


async def _close_screen(message: Message, state: FSMContext, key: str) -> None:
    """Удаляет экран с кнопками и забывает его id, чтобы не править пустое место."""
    try:
        await message.delete()
    except TelegramBadRequest as exc:
        logger.warning("Не удалось удалить экран: %s", exc)
    await _forget_message_screens(state, message.message_id, key)


async def _edit_inline(
    message: Message, text: str, markup: InlineKeyboardMarkup | None
) -> None:
    """Правит текст и кнопки. Повтор того же текста Telegram не считает ошибкой."""
    try:
        await message.edit_text(text, reply_markup=markup)
    except TelegramBadRequest as exc:
        if "message is not modified" not in str(exc).lower():
            logger.warning("Не удалось обновить сообщение: %s", exc)


async def _drop_flow(state: FSMContext) -> None:
    """Выходит из пошагового ввода и сохраняет id уже открытых экранов."""
    data = await state.get_data()
    kept = {key: value for key, value in data.items() if key not in _FLOW_KEYS}
    await state.set_state(None)
    await state.set_data(kept)


async def _upsert_screen(
    message: Message,
    state: FSMContext,
    key: str,
    text: str,
    markup: InlineKeyboardMarkup,
) -> None:
    """Повторный вызов экрана правит прежнее сообщение, иначе отправляет новое."""
    data = await state.get_data()
    message_id = data.get(key)
    bot = message.bot
    if isinstance(message_id, int) and bot is not None:
        try:
            await bot.edit_message_text(
                text,
                chat_id=message.chat.id,
                message_id=message_id,
                reply_markup=markup,
            )
            return
        except TelegramBadRequest as exc:
            lowered = str(exc).lower()
            if "message is not modified" in lowered:
                return
            logger.warning("Не удалось обновить экран: %s", exc)
    sent = await message.answer(text, reply_markup=markup)
    await state.update_data({key: sent.message_id})


async def _assistant_screen(
    chat_id: int, storage: AssistantStorage
) -> tuple[str, InlineKeyboardMarkup]:
    """Собирает экран персон из хранилища."""
    profiles = await storage.list_assistants(chat_id)
    active = await storage.get_active_assistant(chat_id)
    active_id = active.id if active is not None else ""
    return screens._assistants_markup(profiles, active_id)


async def _open_assistants(
    message: Message, storage: AssistantStorage, state: FSMContext
) -> None:
    """Открывает или обновляет экран персон."""
    try:
        text, markup = await _assistant_screen(message.chat.id, storage)
    except AssistantStorageError:
        await _reply_profile_error(message)
        return
    await _upsert_screen(message, state, _SCREEN_ASSISTANTS, text, markup)


async def _topics_screen(
    chat_id: int,
    assistant_id: str,
    storage: TopicHistoryStorage,
) -> tuple[str, InlineKeyboardMarkup]:
    """Собирает экран тем активной персоны."""
    topics = await storage.list_topics(chat_id, assistant_id)
    active = await storage.get_active_topic(chat_id, assistant_id)
    return screens._topics_markup(topics, active)


async def _open_topics(
    message: Message,
    storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Открывает или обновляет экран тем активной персоны."""
    try:
        profile = await _active_profile(message.chat.id, assistant_storage)
        text, markup = await _topics_screen(message.chat.id, profile.id, storage)
    except AssistantStorageError:
        await _reply_profile_error(message)
        return
    except TopicStorageError:
        await _reply_history_error(message)
        return
    await _upsert_screen(message, state, _SCREEN_TOPICS, text, markup)


async def _generate_and_send(
    message: Message,
    assistant: AssistAIClient,
    prompt: str,
    model: str | None,
    size: str = "1024x1024",
    *,
    reply_markup: ReplyKeyboardMarkup | None = None,
) -> None:
    """Просит ProxyAPI нарисовать картинку и отправляет её в чат."""
    user_id = message.from_user.id if message.from_user else message.chat.id
    bind_request_context(user_id=user_id, chat_id=message.chat.id)
    note_active_user(user_id)
    logger.info(
        "image_generation",
        model=model or assistant.config.default_image_model,
        size=size,
        prompt_chars=len(prompt),
    )
    set_billing_user(user_id)
    keep = _KEEP_OUTGOING.set(True)
    try:
        progress = await message.answer("Генерирую изображение…")
    finally:
        _KEEP_OUTGOING.reset(keep)
    try:
        await cost_tracker.check_budget(user_id, cost_tracker.estimate_image_cost(1))
        images = await job_queue.generate_image(assistant, prompt, model, size)
        await _delete_bot_message(progress)
        if not images:
            await message.answer(MSG_IMAGE_FAILED, reply_markup=reply_markup)
            return
        for index, image in enumerate(images):
            await _send_generated_image(message, image, index)
    except BudgetExceededError as exc:
        await _delete_bot_message(progress)
        await message.answer(exc.message, reply_markup=reply_markup)
    except AssistAIError as exc:
        logger.error("AssistAI ошибка: %s", exc.message)
        await _delete_bot_message(progress)
        await message.answer(MSG_IMAGE_FAILED, reply_markup=reply_markup)
    except httpx.TimeoutException:
        logger.error("Таймаут генерации изображения")
        await _delete_bot_message(progress)
        await message.answer(MSG_TIMEOUT, reply_markup=reply_markup)
    except Exception:
        logger.exception("Непредвиденная ошибка генерации изображения")
        await _delete_bot_message(progress)
        await message.answer(MSG_INTERNAL, reply_markup=reply_markup)


async def _open_settings(
    message: Message,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Открывает или обновляет корневой экран /settings."""
    try:
        profile = await _active_profile(message.chat.id, assistant_storage)
    except AssistantStorageError:
        await _reply_profile_error(message)
        return
    text, markup = screens._settings_hub_view(profile)
    await _upsert_screen(message, state, _SCREEN_SETTINGS, text, markup)


async def _open_persona_settings(
    message: Message,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Открывает экран параметров активной персоны."""
    try:
        profile = await _active_profile(message.chat.id, assistant_storage)
    except AssistantStorageError:
        await _reply_profile_error(message)
        return
    text, markup = screens._settings_persona_view(profile, assistant)
    await _upsert_screen(message, state, _SCREEN_SETTINGS, text, markup)


async def _refresh_draft_models(
    query: CallbackQuery, state: FSMContext, assistant: AssistAIClient
) -> None:
    """Обновляет текст и кнопки выбора моделей у создаваемой персоны."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    data = await state.get_data()
    await query.answer()
    await _edit_inline(
        message,
        screens._draft_models_text(data, assistant),
        screens._draft_models_keyboard(data),
    )


async def _persona_id_for_query(
    query: CallbackQuery, message: Message, assistant_storage: AssistantStorage
) -> str | None:
    """Id активной персоны. Ошибку профиля показывает во всплывающем окне."""
    try:
        profile = await _active_profile(message.chat.id, assistant_storage)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return None
    return profile.id


async def _finish_topic_delete(
    chat_id: int,
    assistant_id: str,
    topic_id: str,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
) -> tuple[str, InlineKeyboardMarkup, ReplyKeyboardMarkup]:
    """Удаляет тему. Пустой каталог получает «Общая», меню берёт её имя."""
    await topic_storage.delete_topic(chat_id, assistant_id, topic_id)
    if not await topic_storage.get_active_topic(chat_id, assistant_id):
        await topic_storage.ensure_active(chat_id, assistant_id, "Общая")
    text, markup = await _topics_screen(chat_id, assistant_id, topic_storage)
    menu = await _menu_for(chat_id, assistant_storage, topic_storage)
    return text, markup, menu


async def _refresh_image_screen(
    query: CallbackQuery, state: FSMContext, assistant: AssistAIClient
) -> None:
    """Обновляет экран картинки после выбора модели или размера."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    data = await state.get_data()
    selected = M._optional_model(data.get("image_model"))
    model_id = selected or assistant.config.default_image_model
    size = data.get("image_size")
    allowed = M._size_choices(model_id)
    if not isinstance(size, str) or size not in allowed:
        size = M._default_image_size(model_id)
        await state.update_data(image_size=size)
    await state.set_state(ImageSG.config)
    await query.answer()
    text, markup = screens._image_view(selected, size, assistant)
    await _edit_inline(message, text, markup)
    await state.update_data({_SCREEN_IMAGE: message.message_id})


async def _open_model_menu(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    state: FSMContext,
    *,
    kind: str,
) -> None:
    """Открывает выбор модели отдельным сообщением. Настройки остаются выше."""
    message = _query_message(query)
    if message is None:
        await query.answer()
        return
    try:
        profile = await _active_profile(message.chat.id, assistant_storage)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    await state.update_data({_SCREEN_SETTINGS: message.message_id})
    text, markup = screens._model_picker_parts(profile, assistant, kind)
    await query.answer()
    sent = await message.answer(text, reply_markup=markup)
    await state.update_data({_SCREEN_PICKER: sent.message_id})
