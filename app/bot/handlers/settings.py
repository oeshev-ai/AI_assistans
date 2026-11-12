"""Aiogram handlers."""

from __future__ import annotations

import app.bot.screens as screens
from app.bot import models as M
from app.bot import runtime as R
from app.bot.constants import (
    _SCREEN_PICKER,
    _SCREEN_PRESETS,
    _SCREEN_SETTINGS,
    _VOICE_MODE_LABELS,
    AVAILABLE_IMAGE_MODELS,
    AVAILABLE_TEXT_MODELS,
    AVAILABLE_TTS_MODELS,
    Any,
    AssistAIClient,
    AssistantStorage,
    AssistantStorageError,
    CallbackQuery,
    FSMContext,
    Message,
    TelegramBadRequest,
    is_speechkit_tts_model,
    logger,
    resolve_speechkit_role,
    speechkit_roles_for,
)


async def on_set_text_model(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Записывает текстовую модель в активный профиль."""
    await R._apply_model_choice(
        query,
        assistant,
        assistant_storage,
        state,
        catalog=AVAILABLE_TEXT_MODELS,
        prefix="set_text:",
        kind="text",
    )


async def on_set_image_model(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Записывает модель изображений в активный профиль."""
    await R._apply_model_choice(
        query,
        assistant,
        assistant_storage,
        state,
        catalog=AVAILABLE_IMAGE_MODELS,
        prefix="set_image:",
        kind="image",
    )


async def on_menu_settings(
    message: Message,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """``/settings`` — хаб: настройка, создание, каталог."""
    await R._drop_flow(state)
    await R._open_settings(message, assistant, assistant_storage, state)


async def on_set_text_menu(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Открывает выбор текстовой модели отдельным сообщением."""
    await R._open_model_menu(query, assistant, assistant_storage, state, kind="text")


async def on_set_image_menu(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Открывает выбор модели картинок отдельным сообщением."""
    await R._open_model_menu(query, assistant, assistant_storage, state, kind="image")


async def on_set_tts_model(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Записывает TTS-модель в активный профиль (экран голосового ответа)."""
    await R._apply_model_choice(
        query,
        assistant,
        assistant_storage,
        state,
        catalog=AVAILABLE_TTS_MODELS,
        prefix="set_tts:",
        kind="tts",
    )


async def on_settings_open(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Возвращает сообщение к корневому экрану /settings."""
    message = R._query_message(query)
    if message is None:
        await query.answer()
        return
    try:
        profile = await R._active_profile(message.chat.id, assistant_storage)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    await query.answer()
    text, markup = screens._settings_hub_view(profile)
    await R._edit_inline(message, text, markup)
    data = await state.get_data()
    data[_SCREEN_SETTINGS] = message.message_id
    data.pop(_SCREEN_PRESETS, None)
    await state.set_data(data)


async def on_settings_persona(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Открывает экран настройки активной персоны."""
    message = R._query_message(query)
    if message is None:
        await query.answer()
        return
    try:
        profile = await R._active_profile(message.chat.id, assistant_storage)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    await query.answer()
    text, markup = screens._settings_persona_view(profile, assistant)
    await R._edit_inline(message, text, markup)
    await state.update_data({_SCREEN_SETTINGS: message.message_id})


async def on_settings_close(query: CallbackQuery, state: FSMContext) -> None:
    """Убирает настройки и открытый список моделей."""
    message = R._query_message(query)
    if message is None:
        await query.answer()
        return
    await query.answer()
    data = await state.get_data()
    picker_id = data.get(_SCREEN_PICKER)
    bot = message.bot
    if (
        isinstance(picker_id, int)
        and bot is not None
        and picker_id != message.message_id
    ):
        try:
            await bot.delete_message(chat_id=message.chat.id, message_id=picker_id)
        except TelegramBadRequest as exc:
            logger.warning("Не удалось закрыть выбор модели: %s", exc)
        data.pop(_SCREEN_PICKER, None)
        await state.set_data(data)
    await R._close_screen(message, state, _SCREEN_SETTINGS)


async def on_set_persona_size(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
) -> None:
    """Записывает размер картинки активной персоны (экран выбора модели)."""
    message = R._query_message(query)
    if message is None:
        await query.answer()
        return
    size = (query.data or "").removeprefix("set_size:")
    try:
        chat_id = message.chat.id
        profile = await R._active_profile(chat_id, assistant_storage)
        model_id = M._image_model_of(profile, assistant, chat_id=chat_id)
        if size not in M._size_choices(model_id):
            await query.answer("Этот размер не подходит модели", show_alert=True)
            return
        updated = profile.model_copy(update={"image_size": size})
        await assistant_storage.save_assistant(chat_id, updated)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    await query.answer("Размер обновлён")
    text, markup = screens._model_picker_parts(updated, assistant, "image")
    await R._edit_or_answer(query, text, markup)


async def on_picker_done(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Закрывает выбор модели (картинки / TTS) и возвращает к настройкам."""
    message = R._query_message(query)
    if message is None:
        await query.answer()
        return
    try:
        profile = await R._active_profile(message.chat.id, assistant_storage)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    await query.answer()
    await R._return_to_settings(message, state, profile, assistant)


async def on_voice_menu(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Открывает выбор TTS-модели и голоса отдельным сообщением."""
    await R._open_model_menu(query, assistant, assistant_storage, state, kind="tts")


async def on_voice_mode(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
) -> None:
    """Переключает voice_mode активной персоны."""
    message = R._query_message(query)
    if message is None:
        await query.answer()
        return
    mode = (query.data or "").removeprefix("voice_mode:")
    if mode not in {"off", "auto", "always"}:
        await query.answer("Неизвестный режим", show_alert=True)
        return
    try:
        profile = await R._active_profile(message.chat.id, assistant_storage)
        updated = profile.model_copy(update={"voice_mode": mode})
        await assistant_storage.save_assistant(message.chat.id, updated)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    label = _VOICE_MODE_LABELS[mode]
    await query.answer(f"Режим: {label}")
    text, markup = screens._model_picker_parts(updated, assistant, "tts")
    await R._edit_or_answer(query, text, markup)


async def on_voice_name(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
) -> None:
    """Выбирает голос TTS активной персоны."""
    message = R._query_message(query)
    if message is None:
        await query.answer()
        return
    raw = (query.data or "").removeprefix("voice_name:")
    try:
        profile = await R._active_profile(message.chat.id, assistant_storage)
        voices = M._tts_voices_for(
            assistant, model_id=M._tts_model_of(profile, assistant)
        )
        voice = voices[int(raw)]
        fields: dict[str, Any] = {"tts_voice": voice}
        if is_speechkit_tts_model(M._tts_model_of(profile, assistant)):
            # Сбрасываем амплуа, если новый голос его не поддерживает.
            fields["tts_role"] = resolve_speechkit_role(voice, profile.tts_role) or None
        updated = profile.model_copy(update=fields)
        await assistant_storage.save_assistant(message.chat.id, updated)
    except (ValueError, IndexError):
        await query.answer("Неизвестный голос", show_alert=True)
        return
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    await query.answer(f"Голос: {voice}")
    text, markup = screens._model_picker_parts(updated, assistant, "tts")
    await R._edit_or_answer(query, text, markup)


async def on_voice_role(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
) -> None:
    """Выбирает амплуа SpeechKit для текущего голоса."""
    message = R._query_message(query)
    if message is None:
        await query.answer()
        return
    role = (query.data or "").removeprefix("voice_role:").strip().casefold()
    try:
        profile = await R._active_profile(message.chat.id, assistant_storage)
        voice = M._resolve_profile_tts_voice(profile, assistant)
        if role not in speechkit_roles_for(voice):
            await query.answer("Это амплуа недоступно для голоса", show_alert=True)
            return
        updated = profile.model_copy(update={"tts_role": role})
        await assistant_storage.save_assistant(message.chat.id, updated)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    await query.answer(f"Амплуа: {role}")
    text, markup = screens._model_picker_parts(updated, assistant, "tts")
    await R._edit_or_answer(query, text, markup)
