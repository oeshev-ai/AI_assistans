"""Aiogram handlers."""

from __future__ import annotations

import app.bot.screens as screens
from app.bot import runtime as R
from app.bot.constants import *
from app.bot.states import (
    NewAssistantSG,
)


async def on_presets_open(query: CallbackQuery, state: FSMContext) -> None:
    """Открывает каталог в том же сообщении, что и /settings — без дублей."""
    message = R._query_message(query)
    if message is None:
        await query.answer()
        return
    await query.answer()
    text = screens._presets_text(0)
    markup = screens._presets_keyboard(0)
    await R._edit_inline(message, text, markup)
    await state.update_data(
        {
            _SCREEN_PRESETS: message.message_id,
            _SCREEN_SETTINGS: message.message_id,
        }
    )


async def on_presets_page(query: CallbackQuery, state: FSMContext) -> None:
    """Листает каталог персон."""
    message = R._query_message(query)
    if message is None:
        await query.answer()
        return
    raw = (query.data or "").removeprefix("preset_page:")
    try:
        page = int(raw)
    except ValueError:
        await query.answer("Неизвестная страница", show_alert=True)
        return
    await query.answer()
    text = screens._presets_text(page)
    markup = screens._presets_keyboard(page)
    await R._edit_inline(message, text, markup)
    await state.update_data(
        {
            _SCREEN_PRESETS: message.message_id,
            _SCREEN_SETTINGS: message.message_id,
        }
    )


async def on_preset_view(
    query: CallbackQuery, assistant: AssistAIClient, state: FSMContext
) -> None:
    """Показывает предпросмотр пресета."""
    message = R._query_message(query)
    if message is None:
        await query.answer()
        return
    preset_id = (query.data or "").removeprefix("preset:view:")
    preset = get_preset(preset_id)
    if preset is None:
        await query.answer("Пресет не найден", show_alert=True)
        return
    await query.answer()
    text, markup = screens._preset_preview_parts(preset, assistant)
    await R._edit_inline(message, text, markup)
    await state.update_data(
        {
            _SCREEN_PRESETS: message.message_id,
            _SCREEN_SETTINGS: message.message_id,
        }
    )


async def on_preset_apply(
    query: CallbackQuery,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    topic_storage: TopicHistoryStorage,
    state: FSMContext,
) -> None:
    """Создаёт персону из пресета и делает её активной."""
    message = R._query_message(query)
    if message is None:
        await query.answer()
        return
    preset_id = (query.data or "").removeprefix("preset:apply:")
    preset = get_preset(preset_id)
    if preset is None:
        await query.answer("Пресет не найден", show_alert=True)
        return
    chat_id = message.chat.id
    try:
        profiles = await assistant_storage.list_assistants(chat_id)
        taken = [item.name for item in profiles]
        profile = preset_profile(preset, taken)
        await assistant_storage.save_assistant(chat_id, profile)
        await assistant_storage.set_active_assistant(chat_id, profile.id)
        menu = await R._menu_for(chat_id, assistant_storage, topic_storage)
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    except TopicStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    await R._reset_flow(state)
    await query.answer("Персона создана")
    await R._close_screen(message, state, _SCREEN_PRESETS)
    await R._apply_reply_menu(message, menu, state)
    label = persona_label(profile.name, profile.icon)
    await message.answer(f"✅ Персона «{label}» создана и активна.")
    logger.info(
        "preset_applied",
        chat_id=chat_id,
        preset=preset.id,
        assistant_id=profile.id,
    )


async def on_preset_edit(query: CallbackQuery, state: FSMContext) -> None:
    """Создание из пресета с правкой промпта: правит то же сообщение каталога."""
    message = R._query_message(query)
    if message is None:
        await query.answer()
        return
    preset_id = (query.data or "").removeprefix("preset:edit:")
    preset = get_preset(preset_id)
    if preset is None:
        await query.answer("Пресет не найден", show_alert=True)
        return
    await R._drop_flow(state)
    await state.set_state(NewAssistantSG.prompt)
    data = await state.get_data()
    data.pop(_SCREEN_PRESETS, None)
    data[_SCREEN_SETTINGS] = message.message_id
    data.update(
        {
            "flow_name": preset.name,
            "flow_prompt": preset.system_prompt,
            "flow_icon": normalize_icon(preset.icon),
            "flow_text_model": preset.text_model,
            "flow_image_model": preset.image_model,
            "flow_preset_id": preset.id,
            "flow_edit_id": None,
        }
    )
    await state.set_data(data)
    await query.answer()
    keep = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Оставить как есть",
                    callback_data="asst:keep_prompt",
                )
            ]
        ]
    )
    await R._edit_inline(
        message,
        f"Промпт пресета «{persona_label(preset.name, preset.icon)}»:\n\n"
        f"{preset.system_prompt}\n\n"
        "Пришлите свой промпт (≥10 символов) или оставьте пресетный.\n"
        "/cancel — отмена",
        keep,
    )
