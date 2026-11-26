"""Aiogram handlers."""

from __future__ import annotations

from typing import Any

import app.bot.screens as screens
from app.bot import runtime as R
from app.bot.constants import (
    _SCREEN_PRESETS,
    _SCREEN_SETTINGS,
    AssistAIClient,
    AssistantStorage,
    AssistantStorageError,
    CallbackQuery,
    FSMContext,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    TopicHistoryStorage,
    TopicStorageError,
    get_preset,
    logger,
    normalize_icon,
    persona_label,
    preset_profile,
)
from app.bot.states import (
    NewAssistantSG,
)
from app.storage.assistant_presets import AssistantPreset
from app.storage.assistant_storage import AssistantProfile


def _applied_assistant_id(
    data: dict[str, Any], message_id: int, preset_id: str
) -> str | None:
    """Id персоны, уже созданной этой кнопкой. Иначе ``None``."""
    applied = data.get("preset_apply")
    if not isinstance(applied, dict):
        return None
    if applied.get("message_id") != message_id or applied.get("preset_id") != preset_id:
        return None
    assistant_id = applied.get("assistant_id")
    if isinstance(assistant_id, str) and assistant_id.strip():
        return assistant_id.strip()
    return None


def _valid_profile_id(value: object) -> str | None:
    if isinstance(value, str) and 1 <= len(value.strip()) <= 32:
        return value.strip()
    return None


def _pinned_profile(
    preset: AssistantPreset, profile_id: str, name: str
) -> AssistantProfile:
    """Тот же пресет с уже выбранными id и именем, без нового « (2)»."""
    return preset_profile(preset, []).model_copy(
        update={"id": profile_id, "name": name}
    )


def _preset_pending_markup(preset_id: str) -> InlineKeyboardMarkup:
    """Превью без «Создать»: повторный тап не шлёт второй callback."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="✏️ Создать и настроить",
                    callback_data=f"preset:edit:{preset_id}",
                )
            ],
            [
                InlineKeyboardButton(
                    text="← К каталогу",
                    callback_data="preset:list",
                )
            ],
        ]
    )


async def _claim_preset_profile(
    state: FSMContext,
    preset: AssistantPreset,
    profiles: list[AssistantProfile],
    message_id: int,
) -> AssistantProfile | None:
    """Профиль для записи. ``None`` — это сообщение уже создало персону.

    Id запоминается до save: повторная доставка пишет ту же персону,
    а не «Имя (2)».
    """
    data = await state.get_data()
    applied_id = _applied_assistant_id(data, message_id, preset.id)
    if applied_id is not None:
        return None
    stored_id = _valid_profile_id(data.get("flow_id"))
    stored_name = data.get("flow_name")
    if data.get("flow_preset_id") == preset.id and stored_id is not None:
        existing = next((item for item in profiles if item.id == stored_id), None)
        if existing is not None:
            return existing
        if isinstance(stored_name, str) and stored_name.strip():
            return _pinned_profile(preset, stored_id, stored_name.strip())
    profile = preset_profile(preset, [item.name for item in profiles])
    # До записи: иначе второй callback увидит занятое имя и добавит « (2)».
    await state.update_data(
        flow_id=profile.id,
        flow_preset_id=preset.id,
        flow_name=profile.name,
    )
    fresh = await state.get_data()
    if _applied_assistant_id(fresh, message_id, preset.id) is not None:
        return None
    winner_id = _valid_profile_id(fresh.get("flow_id"))
    winner_name = fresh.get("flow_name")
    if (
        winner_id is not None
        and winner_id != profile.id
        and isinstance(winner_name, str)
        and winner_name.strip()
    ):
        existing = next((item for item in profiles if item.id == winner_id), None)
        if existing is not None:
            return existing
        return _pinned_profile(preset, winner_id, winner_name.strip())
    return profile


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
    """Создаёт персону из пресета. Повтор «Создать» пишет тот же id."""
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
    if _applied_assistant_id(await state.get_data(), message.message_id, preset.id):
        await query.answer("Персона создана")
        return
    if await state.get_state() is not None:
        await query.answer(
            "Сначала завершите текущий шаг или нажмите /cancel.",
            show_alert=True,
        )
        return
    text, _markup = screens._preset_preview_parts(preset, assistant)
    await R._edit_inline(message, text, _preset_pending_markup(preset.id))
    # update_data читает и пишет state отдельно. Лок чата не пускает второй
    # callback в claim, пока первый не сохранил персону и метку preset_apply:
    # _reset_flow стирает flow_id, и без метки ожидание снова создало бы персону.
    try:
        async with R._chat_turn(chat_id):
            profiles = await assistant_storage.list_assistants(chat_id)
            profile = await _claim_preset_profile(
                state, preset, profiles, message.message_id
            )
            if profile is not None:
                await assistant_storage.save_assistant(chat_id, profile)
                await assistant_storage.set_active_assistant(chat_id, profile.id)
                menu = await R._menu_for(chat_id, assistant_storage, topic_storage)
                await R._reset_flow(state)
                await state.update_data(
                    preset_apply={
                        "message_id": message.message_id,
                        "preset_id": preset.id,
                        "assistant_id": profile.id,
                    }
                )
    except AssistantStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    except TopicStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    if profile is None:
        await query.answer("Персона создана")
        return
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
