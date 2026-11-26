"""Aiogram handlers."""

from __future__ import annotations

import app.bot.screens as screens
from app.bot import models as M
from app.bot import runtime as R
from app.bot.constants import (
    _SCREEN_TOPICS,
    AssistantStorage,
    AssistantStorageError,
    CallbackQuery,
    FSMContext,
    Message,
    TopicHistoryStorage,
    TopicStorageError,
    logger,
)
from app.bot.states import (
    NewTopicSG,
    RenameTopicSG,
)


def _stored_topic_name(
    topics: list[dict[str, str]], topic_id: str, fallback: str
) -> str:
    for item in topics:
        if item.get("id") == topic_id:
            return item["name"]
    return fallback


def _persona_suffix(data: str, prefix: str) -> str | None:
    """``prefix`` или ``prefix:{id}``. Пустой хвост — старая кнопка без id."""
    if data == prefix:
        return None
    head = f"{prefix}:"
    if not data.startswith(head):
        return None
    assistant_id = data[len(head) :]
    return assistant_id or None


async def _bound_assistant_id(
    query: CallbackQuery,
    message: Message,
    assistant_storage: AssistantStorage,
    assistant_id: str | None,
) -> str | None:
    """Id с кнопки. Старые сообщения без id берут активную персону."""
    if assistant_id:
        return assistant_id
    return await R._persona_id_for_query(query, message, assistant_storage)


async def on_menu_topics(
    message: Message,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Кнопка «Темы» открывает темы активной персоны."""
    await R._drop_flow(state)
    await R._open_topics(message, topic_storage, assistant_storage, state)


async def on_topics_page(
    query: CallbackQuery,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
) -> None:
    """Открывает другую страницу списка тем."""
    message = query.message
    if not isinstance(message, Message):
        await query.answer()
        return
    raw = (query.data or "").removeprefix("topic_page:")
    bound_id, page_raw = M._callback_persona(raw)
    try:
        page = int(page_raw)
    except ValueError:
        await query.answer("Неверная страница", show_alert=True)
        return
    assistant_id = await _bound_assistant_id(
        query, message, assistant_storage, bound_id
    )
    if assistant_id is None:
        return
    try:
        topics = await topic_storage.list_topics(message.chat.id, assistant_id)
        active = await topic_storage.get_active_topic(message.chat.id, assistant_id)
    except TopicStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    await query.answer()
    await R._edit_or_answer(
        query,
        screens._topics_text(topics, active, page),
        screens._topics_keyboard(topics, active, page, assistant_id=assistant_id),
    )


async def on_topic_delete_ask(
    query: CallbackQuery,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
) -> None:
    """Спрашивает удаление темы в том же сообщении списка."""
    message = query.message
    if not isinstance(message, Message):
        await query.answer()
        return
    bound_id, topic_id = M._callback_persona(
        (query.data or "").removeprefix("topic_del_ask:")
    )
    if not topic_id:
        await query.answer("Тема не найдена.", show_alert=True)
        return
    assistant_id = await _bound_assistant_id(
        query, message, assistant_storage, bound_id
    )
    if assistant_id is None:
        return
    try:
        topics = await topic_storage.list_topics(message.chat.id, assistant_id)
    except TopicStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    name = next((item["name"] for item in topics if item["id"] == topic_id), None)
    if name is None:
        await query.answer("Тема не найдена.", show_alert=True)
        return
    await query.answer()
    await R._edit_inline(
        message,
        f"Удалить тему «{name}» со всей историей?",
        M._confirm_row(
            "✅ Да, удалить",
            f"topic_del_yes:{assistant_id}:{topic_id}",
            f"topic:list:{assistant_id}",
        ),
    )


async def on_topic_delete_yes(
    query: CallbackQuery,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Удаляет тему и обновляет тот же список.

    Нижнее меню переставляется только у активной персоны. Тема другой
    персоны подпись кнопок не меняет.
    """
    message = query.message
    if not isinstance(message, Message):
        await query.answer()
        return
    raw = (query.data or "").removeprefix("topic_del_yes:")
    chat_id = message.chat.id
    # assistant_id:topic_id — персона из UI, не текущая активная.
    bound_id, topic_id = M._callback_persona(raw)
    assistant_id = await _bound_assistant_id(
        query, message, assistant_storage, bound_id
    )
    if assistant_id is None:
        return
    if not topic_id:
        await query.answer("Тема не найдена.", show_alert=True)
        return
    try:
        text, markup, menu = await R._finish_topic_delete(
            chat_id, assistant_id, topic_id, topic_storage, assistant_storage
        )
    except TopicStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    logger.info("Удалена тема chat_id=%s topic_id=%s", chat_id, topic_id)
    await query.answer("Тема удалена")
    await R._edit_inline(message, text, markup)
    await state.update_data({_SCREEN_TOPICS: message.message_id})
    if menu is not None:
        await R._apply_reply_menu(message, menu, state)


async def on_topic_list(
    query: CallbackQuery,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Возвращает сообщение к списку тем."""
    message = R._query_message(query)
    if message is None:
        await query.answer()
        return
    assistant_id = await _bound_assistant_id(
        query,
        message,
        assistant_storage,
        _persona_suffix(query.data or "", "topic:list"),
    )
    if assistant_id is None:
        return
    try:
        text, markup = await R._topics_screen(
            message.chat.id, assistant_id, topic_storage
        )
    except TopicStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    await R._edit_inline(message, text, markup)
    await state.update_data({_SCREEN_TOPICS: message.message_id})
    await query.answer()


async def on_topics_done(query: CallbackQuery, state: FSMContext) -> None:
    """Убирает список тем. Нижнее меню не трогает: активная тема та же."""
    message = R._query_message(query)
    if message is None:
        await query.answer()
        return
    await query.answer()
    await R._close_screen(message, state, _SCREEN_TOPICS)


async def on_topic_use(
    query: CallbackQuery,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Делает тему активной, убирает экран и обновляет нижнее меню."""
    message = R._query_message(query)
    if message is None:
        await query.answer()
        return
    bound_id, topic_id = M._callback_persona(
        (query.data or "").removeprefix("topic:use:")
    )
    if not topic_id:
        await query.answer("Тема не найдена.", show_alert=True)
        return
    chat_id = message.chat.id
    assistant_id = await _bound_assistant_id(
        query, message, assistant_storage, bound_id
    )
    if assistant_id is None:
        return
    try:
        await topic_storage.set_active_topic(chat_id, assistant_id, topic_id)
        menu = await R._menu_for(chat_id, assistant_storage, topic_storage)
    except TopicStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    logger.info("Переключение темы chat_id=%s topic_id=%s", chat_id, topic_id)
    await query.answer()
    await R._close_screen(message, state, _SCREEN_TOPICS)
    await R._apply_reply_menu(message, menu, state)


async def on_topic_rename(
    query: CallbackQuery,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Запрашивает новое название темы."""
    message = R._query_message(query)
    if message is None:
        await query.answer()
        return
    bound_id, topic_id = M._callback_persona(
        (query.data or "").removeprefix("topic:rename:")
    )
    if not topic_id:
        await query.answer("Тема не найдена.", show_alert=True)
        return
    assistant_id = await _bound_assistant_id(
        query, message, assistant_storage, bound_id
    )
    if assistant_id is None:
        return
    try:
        topics = await topic_storage.list_topics(message.chat.id, assistant_id)
    except TopicStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    name = next((item["name"] for item in topics if item["id"] == topic_id), None)
    if name is None:
        await query.answer("Тема не найдена.", show_alert=True)
        return
    await R._drop_flow(state)
    await state.set_state(RenameTopicSG.name)
    await state.update_data(
        flow_topic_id=topic_id,
        flow_topic_name=name,
        flow_assistant_id=assistant_id,
    )
    await query.answer()
    await message.answer(f"Введите новое название для темы «{name}».\n/cancel — отмена")


async def on_topic_new(
    query: CallbackQuery,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Запрашивает название новой темы."""
    message = R._query_message(query)
    if message is None:
        await query.answer()
        return
    assistant_id = await _bound_assistant_id(
        query,
        message,
        assistant_storage,
        _persona_suffix(query.data or "", "topic:new"),
    )
    if assistant_id is None:
        return
    await R._drop_flow(state)
    await state.set_state(NewTopicSG.name)
    await state.update_data(flow_assistant_id=assistant_id)
    await query.answer()
    await message.answer("Введите название темы (до 100 символов).\n/cancel — отмена")


async def sg_new_topic_name(
    message: Message,
    state: FSMContext,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
) -> None:
    """Создаёт тему с введённым названием и делает её активной."""
    name = (message.text or "").strip()
    if not name or len(name) > 100:
        await message.answer(
            "Введите название темы (до 100 символов).\n/cancel — отмена"
        )
        return
    chat_id = message.chat.id
    data = await state.get_data()
    assistant_id = data.get("flow_assistant_id")
    if not isinstance(assistant_id, str) or not assistant_id:
        await R._reset_flow(state)
        menu = await R._menu_for(chat_id, assistant_storage, topic_storage)
        await message.answer("Персона для новой темы не выбрана.", reply_markup=menu)
        return
    try:
        topic_id = await topic_storage.create_topic(chat_id, assistant_id, name)
        await topic_storage.set_active_topic(chat_id, assistant_id, topic_id)
        stored = _stored_topic_name(
            await topic_storage.list_topics(chat_id, assistant_id), topic_id, name
        )
        menu = await R._menu_for(chat_id, assistant_storage, topic_storage)
    except AssistantStorageError:
        await R._reply_profile_error(message)
        return
    except TopicStorageError:
        await R._reply_history_error(message)
        return
    await R._reset_flow(state)
    logger.info("Новая тема chat_id=%s topic_id=%s", chat_id, topic_id)
    await message.answer(
        f"✅ Создана новая тема: '{stored}'. Контекст обнулен.",
        reply_markup=menu,
    )


async def sg_rename_topic_name(
    message: Message,
    state: FSMContext,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
) -> None:
    """Записывает новое название выбранной темы.

    Без id темы или персоны шаг сбрасывается.
    """
    name = (message.text or "").strip()
    if not name or len(name) > 100:
        await message.answer(
            "Введите новое название (до 100 символов).\n/cancel — отмена"
        )
        return
    data = await state.get_data()
    topic_id = data.get("flow_topic_id")
    assistant_id = data.get("flow_assistant_id")
    chat_id = message.chat.id
    if not isinstance(topic_id, str) or not topic_id:
        await R._reset_flow(state)
        menu = await R._menu_for(chat_id, assistant_storage, topic_storage)
        await message.answer("Тема для переименования не выбрана.", reply_markup=menu)
        return
    if not isinstance(assistant_id, str) or not assistant_id:
        await R._reset_flow(state)
        menu = await R._menu_for(chat_id, assistant_storage, topic_storage)
        await message.answer(
            "Персона для переименования не выбрана.", reply_markup=menu
        )
        return
    try:
        await topic_storage.rename_topic(chat_id, assistant_id, topic_id, name)
        stored = _stored_topic_name(
            await topic_storage.list_topics(chat_id, assistant_id), topic_id, name
        )
        menu = await R._menu_for(chat_id, assistant_storage, topic_storage)
    except AssistantStorageError:
        await R._reply_profile_error(message)
        return
    except TopicStorageError:
        await R._reply_history_error(message)
        return
    await R._reset_flow(state)
    logger.info("Тема переименована chat_id=%s topic_id=%s", chat_id, topic_id)
    await R._apply_reply_menu(message, menu, state)
    await message.answer(f"Тема переименована в «{stored}».")
    await R._open_topics(message, topic_storage, assistant_storage, state)
