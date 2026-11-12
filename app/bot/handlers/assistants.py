"""Aiogram handlers."""

from __future__ import annotations

import app.bot.screens as screens
from app.bot import models as M
from app.bot import runtime as R
from app.bot.constants import *
from app.bot.states import (
    EditAssistantSG,
    NewAssistantSG,
)


async def on_menu_assistants(
    message: Message, assistant_storage: AssistantStorage, state: FSMContext
) -> None:
    """Кнопка «Персоны» открывает список персон."""
    await R._drop_flow(state)
    await R._open_assistants(message, assistant_storage, state)


async def on_asst_list(
    query: CallbackQuery, assistant_storage: AssistantStorage, state: FSMContext
) -> None:
    """Возвращает сообщение к списку персон."""
    message = R._query_message(query)
    if message is None:
        await query.answer()
        return
    try:
        text, markup = await R._assistant_screen(message.chat.id, assistant_storage)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    await R._edit_inline(message, text, markup)
    await state.update_data({_SCREEN_ASSISTANTS: message.message_id})
    await query.answer()


async def on_asst_use(
    query: CallbackQuery,
    assistant_storage: AssistantStorage,
    topic_storage: TopicHistoryStorage,
    state: FSMContext,
) -> None:
    """Делает выбранную персону активной, убирает экран и обновляет нижнее меню."""
    message = R._query_message(query)
    if message is None:
        await query.answer()
        return
    profile_id = (query.data or "").removeprefix("asst:use:")
    chat_id = message.chat.id
    try:
        await assistant_storage.set_active_assistant(chat_id, profile_id)
        menu = await R._menu_for(chat_id, assistant_storage, topic_storage)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    except TopicStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    await query.answer()
    await R._close_screen(message, state, _SCREEN_ASSISTANTS)
    await R._apply_reply_menu(message, menu, state)
    logger.info("Активный ассистент chat_id=%s id=%s", chat_id, profile_id)


async def on_assistants_page(
    query: CallbackQuery, assistant_storage: AssistantStorage
) -> None:
    """Открывает другую страницу списка персон."""
    message = query.message
    if not isinstance(message, Message):
        await query.answer()
        return
    try:
        page = int((query.data or "").removeprefix("asst_page:"))
    except ValueError:
        await query.answer("Неверная страница", show_alert=True)
        return
    try:
        profiles = await assistant_storage.list_assistants(message.chat.id)
        active = await assistant_storage.get_active_assistant(message.chat.id)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    active_id = active.id if active is not None else ""
    await query.answer()
    await R._edit_or_answer(
        query,
        screens._assistants_text(profiles, active_id, page),
        screens._assistants_keyboard(profiles, active_id, page),
    )


async def on_asst_rename_ask(
    query: CallbackQuery, state: FSMContext, assistant_storage: AssistantStorage
) -> None:
    """Просит новое имя активной или выбранной персоны."""
    message = query.message
    if not isinstance(message, Message):
        await query.answer()
        return
    profile_id = (query.data or "").removeprefix("asst_rename:")
    try:
        profiles = await assistant_storage.list_assistants(message.chat.id)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    profile = next((item for item in profiles if item.id == profile_id), None)
    if profile is None:
        await query.answer("Ассистент не найден", show_alert=True)
        return
    await state.set_state(EditAssistantSG.name)
    await state.update_data(id=profile.id, name=profile.name)
    await query.answer()
    await R._edit_or_answer(
        query,
        f"Текущее имя: «{persona_label(profile.name, profile.icon)}»\n\n"
        "Введите новое имя (1–50 символов). /cancel — отмена.",
    )


async def sg_edit_name(
    message: Message,
    state: FSMContext,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
) -> None:
    """Сохраняет новое имя выбранной персоны."""
    name = (message.text or "").strip()
    if not name or len(name) > 50:
        await message.answer("Имя: 1–50 символов. /cancel — отмена.")
        return
    data = await state.get_data()
    profile_id = data.get("id")
    if not isinstance(profile_id, str) or not profile_id:
        await R._reset_flow(state)
        await message.answer("Ассистент не найден.")
        return
    try:
        profiles = await assistant_storage.list_assistants(message.chat.id)
        old = next((item for item in profiles if item.id == profile_id), None)
        if old is None:
            await R._reset_flow(state)
            await message.answer("Ассистент не найден.")
            return
        updated = old.model_copy(update={"name": name})
        await assistant_storage.save_assistant(message.chat.id, updated)
    except AssistantStorageError:
        await R._reply_profile_error(message)
        return
    except ValidationError:
        await message.answer("Имя: 1–50 символов. /cancel — отмена.")
        return
    await R._reset_flow(state)
    logger.info(
        "Обновлено имя ассистента chat_id=%s id=%s", message.chat.id, updated.id
    )
    await message.answer(
        f"✅ Имя ассистента изменено на «{persona_label(updated.name, updated.icon)}»."
    )
    await R._open_persona_settings(message, assistant, assistant_storage, state)


async def on_asst_edit_ask(
    query: CallbackQuery, state: FSMContext, assistant_storage: AssistantStorage
) -> None:
    """Показывает текущий промпт и ждёт новый текст."""
    message = query.message
    if not isinstance(message, Message):
        await query.answer()
        return
    profile_id = (query.data or "").removeprefix("asst_edit:")
    try:
        profiles = await assistant_storage.list_assistants(message.chat.id)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    profile = next((item for item in profiles if item.id == profile_id), None)
    if profile is None:
        await query.answer("Ассистент не найден", show_alert=True)
        return
    await state.set_state(EditAssistantSG.prompt)
    await state.update_data(id=profile.id, name=profile.name)
    await query.answer()
    await R._edit_or_answer(
        query,
        f"Промпт ассистента «{persona_label(profile.name, profile.icon)}»:\n\n"
        f"{profile.system_prompt}\n\n"
        "Пришлите новый промпт (≥10 символов). /cancel — отмена.",
    )


async def sg_edit_prompt(
    message: Message,
    state: FSMContext,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
) -> None:
    """Сохраняет новый системный промпт выбранной персоны."""
    prompt = message.text or ""
    if len(prompt) < 10:
        await message.answer("Промпт минимум 10 символов. /cancel — отмена.")
        return
    data = await state.get_data()
    profile_id = data.get("id")
    if not isinstance(profile_id, str) or not profile_id:
        await R._reset_flow(state)
        await message.answer("Ассистент не найден.")
        return
    try:
        profiles = await assistant_storage.list_assistants(message.chat.id)
        old = next((item for item in profiles if item.id == profile_id), None)
        if old is None:
            await R._reset_flow(state)
            await message.answer("Ассистент не найден.")
            return
        updated = old.model_copy(update={"system_prompt": prompt})
        await assistant_storage.save_assistant(message.chat.id, updated)
    except AssistantStorageError:
        await R._reply_profile_error(message)
        return
    except ValidationError:
        await message.answer("Промпт минимум 10 символов. /cancel — отмена.")
        return
    await R._reset_flow(state)
    logger.info(
        "Обновлён промпт ассистента chat_id=%s id=%s", message.chat.id, updated.id
    )
    await message.answer(
        f"✅ Промпт ассистента «{persona_label(updated.name, updated.icon)}» обновлён."
    )
    await R._open_persona_settings(message, assistant, assistant_storage, state)


async def on_asst_edit(
    query: CallbackQuery, assistant_storage: AssistantStorage, state: FSMContext
) -> None:
    """Запускает правку промпта существующей персоны."""
    message = R._query_message(query)
    if message is None:
        await query.answer()
        return
    profile_id = (query.data or "").removeprefix("asst:edit:")
    try:
        profiles = await assistant_storage.list_assistants(message.chat.id)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    profile = next((item for item in profiles if item.id == profile_id), None)
    if profile is None:
        await query.answer("Ассистент не найден.", show_alert=True)
        return
    await R._drop_flow(state)
    await state.set_state(NewAssistantSG.prompt)
    await state.update_data(
        flow_edit_id=profile.id,
        flow_name=profile.name,
        flow_icon=normalize_icon(profile.icon),
        flow_text_model=profile.text_model,
        flow_image_model=profile.image_model,
    )
    await query.answer()
    await message.answer(
        f"Пришлите системный промпт для «{persona_label(profile.name, profile.icon)}» "
        "(мин. 10 символов).\n"
        "/cancel — отмена"
    )


async def on_asst_del_ask(
    query: CallbackQuery, assistant_storage: AssistantStorage
) -> None:
    """Спрашивает подтверждение перед удалением персоны."""
    message = R._query_message(query)
    if message is None:
        await query.answer()
        return
    profile_id = (query.data or "").removeprefix("asst:del_ask:")
    if profile_id == DEFAULT_ASSISTANT_ID:
        await query.answer("Базового ассистента удалить нельзя.", show_alert=True)
        return
    try:
        profiles = await assistant_storage.list_assistants(message.chat.id)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    profile = next((item for item in profiles if item.id == profile_id), None)
    if profile is None:
        await query.answer("Ассистент не найден.", show_alert=True)
        return
    await query.answer()
    await R._edit_inline(
        message,
        f"Удалить персону «{persona_label(profile.name, profile.icon)}»?",
        M._confirm_row(
            "✅ Да, удалить", f"asst:del_yes:{profile.id}", "settings:persona"
        ),
    )


async def on_asst_del_yes(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    topic_storage: TopicHistoryStorage,
    state: FSMContext,
) -> None:
    """Удаляет персону и её темы, затем показывает настройки оставшейся."""
    message = R._query_message(query)
    if message is None:
        await query.answer()
        return
    profile_id = (query.data or "").removeprefix("asst:del_yes:")
    chat_id = message.chat.id
    try:
        profiles = await assistant_storage.list_assistants(chat_id)
        deleted = next((item for item in profiles if item.id == profile_id), None)
        name = (
            persona_label(deleted.name, deleted.icon)
            if deleted is not None
            else profile_id
        )
        if profile_id != DEFAULT_ASSISTANT_ID:
            await topic_storage.delete_assistant_topics(chat_id, profile_id)
        await assistant_storage.delete_assistant(chat_id, profile_id)
        profile = await R._active_profile(chat_id, assistant_storage)
        menu = await R._menu_for(chat_id, assistant_storage, topic_storage)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    except TopicStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    logger.info("Удалён ассистент chat_id=%s id=%s", chat_id, profile_id)
    await query.answer("Удалено")
    text, markup = screens._settings_hub_view(profile)
    await R._edit_inline(message, text, markup)
    await state.update_data({_SCREEN_SETTINGS: message.message_id})
    await message.answer(f"Персона «{name}» удалена.", reply_markup=menu)


async def on_asst_new_start(query: CallbackQuery, state: FSMContext) -> None:
    """Первый шаг создания персоны: запрос имени."""
    message = R._query_message(query)
    if message is None:
        await query.answer()
        return
    await R._drop_flow(state)
    await state.set_state(NewAssistantSG.name)
    await state.update_data(flow_icon=DEFAULT_ICON)
    await query.answer()
    await message.answer("Введите имя персоны (1–50 символов).\n/cancel — отмена")


async def sg_new_assistant_name(message: Message, state: FSMContext) -> None:
    """Принимает имя персоны и переходит к промпту."""
    name = (message.text or "").strip()
    if not name or len(name) > 50:
        await message.answer("Введите имя персоны (1–50 символов).\n/cancel — отмена")
        return
    data = await state.get_data()
    icon = data.get("flow_icon") or DEFAULT_ICON
    await state.update_data(
        flow_name=name, flow_edit_id=None, flow_icon=normalize_icon(str(icon))
    )
    await state.set_state(NewAssistantSG.prompt)
    await message.answer(
        "Пришлите системный промпт (мин. 10 символов).\n/cancel — отмена"
    )


async def sg_new_assistant_prompt(
    message: Message, state: FSMContext, assistant: AssistAIClient
) -> None:
    """Принимает промпт, убирает прошлый экран мастера и показывает модели."""
    prompt = (message.text or "").strip()
    if len(prompt) < 10:
        await message.answer(
            "Пришлите системный промпт (мин. 10 символов).\n/cancel — отмена"
        )
        return
    await state.update_data(flow_prompt=prompt)
    await state.set_state(NewAssistantSG.models)
    data = await state.get_data()
    prev_id = data.get(_SCREEN_SETTINGS)
    bot = message.bot
    if isinstance(prev_id, int) and bot is not None:
        try:
            await bot.delete_message(chat_id=message.chat.id, message_id=prev_id)
        except TelegramBadRequest:
            pass
        await R._forget_message_screens(state, prev_id)
    data = await state.get_data()
    sent = await message.answer(
        screens._draft_models_text(data, assistant),
        reply_markup=screens._draft_models_keyboard(data),
    )
    await state.update_data({_SCREEN_SETTINGS: sent.message_id})


async def sg_new_assistant_models(message: Message) -> None:
    """На шаге моделей текст не уходит в языковую модель."""
    await message.answer(
        "Выберите модели кнопками или нажмите «Готово».\n/cancel — отмена"
    )


async def on_asst_new_text(
    query: CallbackQuery, state: FSMContext, assistant: AssistAIClient
) -> None:
    """Запоминает текстовую модель до нажатия «Готово»."""
    if await state.get_state() != NewAssistantSG.models.state:
        await query.answer("Сначала начните создание персоны.", show_alert=True)
        return
    raw = (query.data or "").removeprefix("asst:new_text:")
    try:
        selected: str | None = AVAILABLE_TEXT_MODELS[int(raw)]
    except (ValueError, IndexError):
        await query.answer("Неизвестная модель", show_alert=True)
        return
    await state.update_data(flow_text_model=selected)
    await R._refresh_draft_models(query, state, assistant)


async def on_asst_new_image(
    query: CallbackQuery, state: FSMContext, assistant: AssistAIClient
) -> None:
    """Запоминает модель картинок до нажатия «Готово»."""
    if await state.get_state() != NewAssistantSG.models.state:
        await query.answer("Сначала начните создание персоны.", show_alert=True)
        return
    raw = (query.data or "").removeprefix("asst:new_image:")
    try:
        selected: str | None = AVAILABLE_IMAGE_MODELS[int(raw)]
    except (ValueError, IndexError):
        await query.answer("Неизвестная модель", show_alert=True)
        return
    await state.update_data(flow_image_model=selected)
    await R._refresh_draft_models(query, state, assistant)


async def on_asst_new_commit(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    topic_storage: TopicHistoryStorage,
    state: FSMContext,
) -> None:
    """Сохраняет персону из шагов мастера и делает её активной."""
    message = R._query_message(query)
    if message is None:
        await query.answer()
        return
    if await state.get_state() != NewAssistantSG.models.state:
        await query.answer("Сначала начните создание персоны.", show_alert=True)
        return
    data = await state.get_data()
    name = data.get("flow_name")
    prompt = data.get("flow_prompt")
    edit_id = data.get("flow_edit_id")
    if not isinstance(name, str) or not isinstance(prompt, str):
        await query.answer("Имя 1–50, промпт ≥ 10", show_alert=True)
        await state.set_state(NewAssistantSG.name)
        await message.answer("Введите имя персоны (1–50 символов).\n/cancel — отмена")
        return
    try:
        fields: dict[str, Any] = {
            "name": name,
            "system_prompt": prompt,
            "icon": normalize_icon(
                data.get("flow_icon")
                if isinstance(data.get("flow_icon"), str)
                else None
            ),
            "text_model": M._optional_model(data.get("flow_text_model")),
            "image_model": M._optional_model(data.get("flow_image_model")),
        }
        if isinstance(edit_id, str) and edit_id:
            fields["id"] = edit_id
        profile = AssistantProfile.model_validate(fields)
    except ValidationError:
        await query.answer("Имя 1–50, промпт ≥ 10", show_alert=True)
        await state.set_state(NewAssistantSG.name)
        await message.answer("Введите имя персоны (1–50 символов).\n/cancel — отмена")
        return
    chat_id = message.chat.id
    try:
        await assistant_storage.save_assistant(chat_id, profile)
        await assistant_storage.set_active_assistant(chat_id, profile.id)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    await R._reset_flow(state)
    logger.info("Сохранён ассистент chat_id=%s id=%s", chat_id, profile.id)
    await query.answer("Сохранено")
    await R._cleanup_wizard_ui(
        message,
        state,
        assistant_storage,
        topic_storage,
        also_delete=[message.message_id],
    )
    verb = "Обновлён" if isinstance(edit_id, str) and edit_id else "Создан"
    # Меню уже на невидимом якоре: не вешаем его на текст, иначе удаление
    # подтверждения снова спрячет кнопки.
    label = persona_label(profile.name, profile.icon)
    await message.answer(f"✅ {verb} ассистент «{label}» ({profile.id}).")
    await R._open_settings(message, assistant, assistant_storage, state)


async def on_asst_icon_menu(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Открывает выбор иконки активной персоны отдельным сообщением."""
    message = R._query_message(query)
    if message is None:
        await query.answer()
        return
    try:
        profile = await R._active_profile(message.chat.id, assistant_storage)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    await state.update_data({_SCREEN_SETTINGS: message.message_id})
    label = persona_label(profile.name, profile.icon)
    await query.answer()
    sent = await message.answer(
        f"Иконка персоны: {label}",
        reply_markup=screens._icons_keyboard("set_icon", profile.icon),
    )
    await state.update_data({_SCREEN_PICKER: sent.message_id})


async def on_set_persona_icon(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Сохраняет иконку активной персоны и возвращает к настройкам."""
    message = R._query_message(query)
    if message is None:
        await query.answer()
        return
    raw = (query.data or "").removeprefix("set_icon:")
    try:
        icon = PERSONA_ICONS[int(raw)]
    except (ValueError, IndexError):
        await query.answer("Неизвестная иконка", show_alert=True)
        return
    try:
        profile = await R._active_profile(message.chat.id, assistant_storage)
        updated = profile.model_copy(update={"icon": icon})
        await assistant_storage.save_assistant(message.chat.id, updated)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    await query.answer("Иконка обновлена")
    await R._return_to_settings(message, state, updated, assistant)


async def on_asst_new_icon_menu(query: CallbackQuery, state: FSMContext) -> None:
    """Пикер иконки на шаге моделей: правит то же сообщение, без новых списков."""
    if await state.get_state() != NewAssistantSG.models.state:
        await query.answer("Сначала начните создание персоны.", show_alert=True)
        return
    message = R._query_message(query)
    if message is None:
        await query.answer()
        return
    data = await state.get_data()
    current = normalize_icon(
        data.get("flow_icon") if isinstance(data.get("flow_icon"), str) else None
    )
    await state.update_data({_SCREEN_SETTINGS: message.message_id})
    await query.answer()
    await R._edit_inline(
        message,
        f"Иконка персоны: {current}\nВыберите эмодзи.",
        screens._icons_keyboard("asst:new_icon", current),
    )


async def on_asst_new_icon(
    query: CallbackQuery, state: FSMContext, assistant: AssistAIClient
) -> None:
    """Запоминает иконку черновика и возвращает один экран моделей на месте."""
    if await state.get_state() != NewAssistantSG.models.state:
        await query.answer("Сначала начните создание персоны.", show_alert=True)
        return
    message = R._query_message(query)
    if message is None:
        await query.answer()
        return
    raw = (query.data or "").removeprefix("asst:new_icon:")
    try:
        icon = PERSONA_ICONS[int(raw)]
    except (ValueError, IndexError):
        await query.answer("Неизвестная иконка", show_alert=True)
        return
    await state.update_data(flow_icon=icon)
    data = await state.get_data()
    await query.answer(f"Иконка: {icon}")
    await R._edit_inline(
        message,
        screens._draft_models_text(data, assistant),
        screens._draft_models_keyboard(data),
    )
    await state.update_data({_SCREEN_SETTINGS: message.message_id})


async def on_asst_keep_prompt(
    query: CallbackQuery, state: FSMContext, assistant: AssistAIClient
) -> None:
    """Оставляет промпт пресета и показывает модели в том же сообщении."""
    if await state.get_state() != NewAssistantSG.prompt.state:
        await query.answer("Сначала начните создание персоны.", show_alert=True)
        return
    message = R._query_message(query)
    if message is None:
        await query.answer()
        return
    data = await state.get_data()
    prompt = data.get("flow_prompt")
    if not isinstance(prompt, str) or len(prompt.strip()) < 10:
        await query.answer("Промпт пресета не найден", show_alert=True)
        return
    await state.set_state(NewAssistantSG.models)
    await query.answer()
    data = await state.get_data()
    await R._edit_inline(
        message,
        screens._draft_models_text(data, assistant),
        screens._draft_models_keyboard(data),
    )
    await state.update_data({_SCREEN_SETTINGS: message.message_id})
