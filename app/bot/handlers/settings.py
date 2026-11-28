"""Aiogram handlers."""

from __future__ import annotations

from pydantic import ValidationError

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
    IMAGE_SIZES,
    Any,
    AssistAIClient,
    AssistantStorage,
    AssistantStorageError,
    CallbackQuery,
    FSMContext,
    Message,
    is_speechkit_tts_model,
    resolve_speechkit_role,
    speechkit_roles_for,
)
from app.llm.video_settings import parse_video_callback, split_video_persona


async def on_set_text_model(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Записывает текстовую модель персоне с кнопки."""
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
    """Записывает модель изображений персоне с кнопки."""
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


async def on_video_menu(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Открывает настройки ролика активной персоны отдельным сообщением."""
    await R._open_model_menu(query, assistant, assistant_storage, state, kind="video")


async def on_set_video(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
) -> None:
    """Записывает формат, длину, разрешение или звук ролика персоны с кнопки."""
    message = R._query_message(query)
    if message is None:
        await query.answer()
        return
    assistant_id, payload = split_video_persona(query.data or "")
    parsed = parse_video_callback(payload)
    if parsed is None:
        await query.answer("Неизвестная настройка", show_alert=True)
        return
    field, value, label = parsed
    profile = await R._profile_for_ui(query, message, assistant_storage, assistant_id)
    if profile is None:
        return
    try:
        updated = profile.validated_update({field: value})
        await assistant_storage.save_assistant(message.chat.id, updated)
    except ValidationError:
        await query.answer("Некорректное значение", show_alert=True)
        return
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    await query.answer(label)
    text, markup = screens._model_picker_parts(updated, assistant, "video")
    await R._edit_or_answer(query, text, markup)


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
    """Записывает TTS-модель персоне с кнопки (экран голосового ответа)."""
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
    await R._dismiss_open_picker(message, state)
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
    await R._dismiss_open_picker(message, state)
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
    await R._dismiss_open_picker(message, state)
    await R._close_screen(message, state, _SCREEN_SETTINGS)


async def on_set_persona_size(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
) -> None:
    """Записывает размер картинки персоны с кнопки (экран выбора модели)."""
    message = R._query_message(query)
    if message is None:
        await query.answer()
        return
    assistant_id, size = M._callback_persona(
        (query.data or "").removeprefix("set_size:"),
        legacy_values=IMAGE_SIZES,
    )
    profile = await R._profile_for_ui(query, message, assistant_storage, assistant_id)
    if profile is None:
        return
    try:
        chat_id = message.chat.id
        model_id = M._image_model_of(profile, assistant, chat_id=chat_id)
        if size not in M._size_choices(model_id):
            await query.answer("Этот размер не подходит модели", show_alert=True)
            return
        updated = profile.validated_update({"image_size": size})
        await assistant_storage.save_assistant(chat_id, updated)
    except ValidationError:
        await query.answer("Некорректное значение", show_alert=True)
        return
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    await query.answer("Размер обновлён")
    text, markup = screens._model_picker_parts(updated, assistant, "image")
    await R._edit_or_answer(query, text, markup)


async def on_history_menu(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Открывает длину истории персоны с кнопки, не активной."""
    message = R._query_message(query)
    if message is None:
        await query.answer()
        return
    raw = (query.data or "").removeprefix("history_menu:")
    assistant_id = raw or None
    profile = await R._profile_for_ui(query, message, assistant_storage, assistant_id)
    if profile is None:
        return
    await state.update_data({_SCREEN_SETTINGS: message.message_id})
    text, markup = screens._model_picker_parts(profile, assistant, "chat")
    await query.answer()
    sent = await message.answer(text, reply_markup=markup)
    await state.update_data({_SCREEN_PICKER: sent.message_id})


async def on_set_history(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
) -> None:
    """Записывает длину истории персоны с кнопки."""
    message = R._query_message(query)
    if message is None:
        await query.answer()
        return
    assistant_id, raw_value = M._callback_persona(
        (query.data or "").removeprefix("set_history:")
    )
    profile = await R._profile_for_ui(query, message, assistant_storage, assistant_id)
    if profile is None:
        return
    if raw_value == "default":
        value: int | None = None
    else:
        try:
            value = int(raw_value)
        except ValueError:
            await query.answer("Некорректное значение", show_alert=True)
            return
    try:
        updated = profile.validated_update({"history_length": value})
        await assistant_storage.save_assistant(message.chat.id, updated)
    except ValidationError:
        await query.answer("Некорректное значение", show_alert=True)
        return
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    await query.answer("История обновлена")
    text, markup = screens._model_picker_parts(updated, assistant, "chat")
    await R._edit_or_answer(query, text, markup)


async def on_set_persona_background(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
) -> None:
    """Записывает фон генерации GPT Image. Правка это поле не использует."""
    message = R._query_message(query)
    if message is None:
        await query.answer()
        return
    assistant_id, background = M._callback_persona(
        (query.data or "").removeprefix("set_bg:")
    )
    if background not in M._BACKGROUND_LABELS:
        await query.answer("Неизвестный фон", show_alert=True)
        return
    profile = await R._profile_for_ui(query, message, assistant_storage, assistant_id)
    if profile is None:
        return
    try:
        chat_id = message.chat.id
        if M._image_background_of(profile, assistant) is None:
            await query.answer("Эта модель не задаёт фон", show_alert=True)
            return
        updated = profile.validated_update({"image_background": background})
        await assistant_storage.save_assistant(chat_id, updated)
    except ValidationError:
        await query.answer("Некорректное значение", show_alert=True)
        return
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    await query.answer("Фон обновлён")
    text, markup = screens._model_picker_parts(updated, assistant, "image")
    await R._edit_or_answer(query, text, markup)


async def on_set_image_file(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
) -> None:
    """Записывает «файлом» для текущей модели картинок. У остальных моделей свой флаг."""
    message = R._query_message(query)
    if message is None:
        await query.answer()
        return
    assistant_id, flag = M._callback_persona(
        (query.data or "").removeprefix("set_file:")
    )
    if flag not in {"0", "1"}:
        await query.answer("Неизвестная настройка", show_alert=True)
        return
    profile = await R._profile_for_ui(query, message, assistant_storage, assistant_id)
    if profile is None:
        return
    try:
        chat_id = message.chat.id
        model_id = M._image_model_of(profile, assistant, chat_id=chat_id)
        flags = dict(profile.image_as_file)
        if flag == "1":
            flags[model_id] = True
        else:
            flags.pop(model_id, None)
        updated = profile.validated_update({"image_as_file": flags})
        await assistant_storage.save_assistant(chat_id, updated)
    except ValidationError:
        await query.answer("Некорректное значение", show_alert=True)
        return
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    await query.answer("Результат: файл" if flag == "1" else "Результат: фото")
    text, markup = screens._model_picker_parts(updated, assistant, "image")
    await R._edit_or_answer(query, text, markup)


async def on_picker_done(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Закрывает выбор модели и возвращает к настройкам персоны с кнопки."""
    message = R._query_message(query)
    if message is None:
        await query.answer()
        return
    suffix = (query.data or "").removeprefix("picker:done")
    assistant_id = suffix[1:] if suffix.startswith(":") and len(suffix) > 1 else None
    profile = await R._profile_for_ui(query, message, assistant_storage, assistant_id)
    if profile is None:
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
    """Переключает voice_mode персоны с кнопки."""
    message = R._query_message(query)
    if message is None:
        await query.answer()
        return
    assistant_id, mode = M._callback_persona(
        (query.data or "").removeprefix("voice_mode:")
    )
    if mode not in {"off", "auto", "always"}:
        await query.answer("Неизвестный режим", show_alert=True)
        return
    profile = await R._profile_for_ui(query, message, assistant_storage, assistant_id)
    if profile is None:
        return
    try:
        updated = profile.validated_update({"voice_mode": mode})
        await assistant_storage.save_assistant(message.chat.id, updated)
    except ValidationError:
        await query.answer("Некорректное значение", show_alert=True)
        return
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
    """Выбирает голос TTS персоны с кнопки."""
    message = R._query_message(query)
    if message is None:
        await query.answer()
        return
    assistant_id, raw = M._callback_persona(
        (query.data or "").removeprefix("voice_name:")
    )
    profile = await R._profile_for_ui(query, message, assistant_storage, assistant_id)
    if profile is None:
        return
    voices = M._tts_voices_for(assistant, model_id=M._tts_model_of(profile, assistant))
    try:
        voice = voices[int(raw)]
    except (ValueError, IndexError):
        await query.answer("Неизвестный голос", show_alert=True)
        return
    fields: dict[str, Any] = {"tts_voice": voice}
    if is_speechkit_tts_model(M._tts_model_of(profile, assistant)):
        # Сбрасываем амплуа, если новый голос его не поддерживает.
        fields["tts_role"] = resolve_speechkit_role(voice, profile.tts_role) or None
    try:
        updated = profile.validated_update(fields)
        await assistant_storage.save_assistant(message.chat.id, updated)
    except ValidationError:
        await query.answer("Некорректное значение", show_alert=True)
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
    assistant_id, role = M._callback_persona(
        (query.data or "").removeprefix("voice_role:")
    )
    role = role.strip().casefold()
    profile = await R._profile_for_ui(query, message, assistant_storage, assistant_id)
    if profile is None:
        return
    try:
        voice = M._resolve_profile_tts_voice(profile, assistant)
        if role not in speechkit_roles_for(voice):
            await query.answer("Это амплуа недоступно для голоса", show_alert=True)
            return
        updated = profile.validated_update({"tts_role": role})
        await assistant_storage.save_assistant(message.chat.id, updated)
    except ValidationError:
        await query.answer("Некорректное значение", show_alert=True)
        return
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    await query.answer(f"Амплуа: {role}")
    text, markup = screens._model_picker_parts(updated, assistant, "tts")
    await R._edit_or_answer(query, text, markup)
