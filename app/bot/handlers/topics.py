"""Aiogram handlers."""

from __future__ import annotations

import app.bot.screens as screens
from app.bot import models as M
from app.bot import runtime as R
from app.bot.constants import *
from app.bot.states import (
    NewTopicSG,
    RenameTopicSG,
)


async def on_menu_topics(
    message: Message,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Кнопка «Темы» открывает темы активной персоны."""
    await R._drop_flow(state)
    await R._open_topics(message, topic_storage, assistant_storage, state)


async def on_topics_refresh(
    query: CallbackQuery,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
) -> None:
    """Перерисовывает список тем с первой страницы в том же сообщении."""
    message = query.message
    if not isinstance(message, Message):
        await query.answer()
        return
    assistant_id = await R._persona_id_for_query(query, message, assistant_storage)
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
        screens._topics_text(topics, active),
        screens._topics_keyboard(topics, active),
    )


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
    try:
        page = int((query.data or "").removeprefix("topic_page:"))
    except ValueError:
        await query.answer("Неверная страница", show_alert=True)
        return
    assistant_id = await R._persona_id_for_query(query, message, assistant_storage)
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
        screens._topics_keyboard(topics, active, page),
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
    topic_id = (query.data or "").removeprefix("topic_del_ask:")
    assistant_id = await R._persona_id_for_query(query, message, assistant_storage)
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
        M._confirm_row("✅ Да, удалить", f"topic_del_yes:{topic_id}", "topic:list"),
    )


async def on_topic_delete_yes(
    query: CallbackQuery,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Удаляет тему и историю, затем обновляет тот же список тем."""
    message = query.message
    if not isinstance(message, Message):
        await query.answer()
        return
    topic_id = (query.data or "").removeprefix("topic_del_yes:")
    chat_id = message.chat.id
    assistant_id = await R._persona_id_for_query(query, message, assistant_storage)
    if assistant_id is None:
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
    assistant_id = await R._persona_id_for_query(query, message, assistant_storage)
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
    topic_id = (query.data or "").removeprefix("topic:use:")
    chat_id = message.chat.id
    assistant_id = await R._persona_id_for_query(query, message, assistant_storage)
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


async def on_topic_clear_ask(
    query: CallbackQuery,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
) -> None:
    """Спрашивает подтверждение очистки истории темы."""
    message = R._query_message(query)
    if message is None:
        await query.answer()
        return
    topic_id = (query.data or "").removeprefix("topic:clear_ask:")
    assistant_id = await R._persona_id_for_query(query, message, assistant_storage)
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
        f"Очистить историю темы «{name}»?",
        M._confirm_row("✅ Да, очистить", f"topic:clear_yes:{topic_id}", "topic:list"),
    )


async def on_topic_clear_yes(
    query: CallbackQuery,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Очищает историю темы и показывает список."""
    message = R._query_message(query)
    if message is None:
        await query.answer()
        return
    topic_id = (query.data or "").removeprefix("topic:clear_yes:")
    chat_id = message.chat.id
    assistant_id = await R._persona_id_for_query(query, message, assistant_storage)
    if assistant_id is None:
        return
    try:
        await topic_storage.clear_topic(chat_id, assistant_id, topic_id)
        text, markup = await R._topics_screen(chat_id, assistant_id, topic_storage)
    except TopicStorageError as exc:
        await query.answer(exc.message, show_alert=True)
        return
    logger.info("Тема очищена chat_id=%s topic_id=%s", chat_id, topic_id)
    await query.answer("История очищена")
    await R._edit_inline(message, text, markup)
    await state.update_data({_SCREEN_TOPICS: message.message_id})


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
    topic_id = (query.data or "").removeprefix("topic:rename:")
    assistant_id = await R._persona_id_for_query(query, message, assistant_storage)
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
    await state.update_data(flow_topic_id=topic_id, flow_topic_name=name)
    await query.answer()
    await message.answer(f"Введите новое название для темы «{name}».\n/cancel — отмена")


async def on_topic_del_ask(
    query: CallbackQuery,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
) -> None:
    """Спрашивает подтверждение удаления темы."""
    message = R._query_message(query)
    if message is None:
        await query.answer()
        return
    topic_id = (query.data or "").removeprefix("topic:del_ask:")
    assistant_id = await R._persona_id_for_query(query, message, assistant_storage)
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
        f"Удалить тему «{name}»?",
        M._confirm_row("✅ Да, удалить", f"topic:del_yes:{topic_id}", "topic:list"),
    )


async def on_topic_del_yes(
    query: CallbackQuery,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    state: FSMContext,
) -> None:
    """Удаляет тему. Если активных не осталось, создаёт тему по умолчанию."""
    message = R._query_message(query)
    if message is None:
        await query.answer()
        return
    topic_id = (query.data or "").removeprefix("topic:del_yes:")
    chat_id = message.chat.id
    assistant_id = await R._persona_id_for_query(query, message, assistant_storage)
    if assistant_id is None:
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
    await R._apply_reply_menu(message, menu, state)


async def on_topic_new(query: CallbackQuery, state: FSMContext) -> None:
    """Запрашивает название новой темы."""
    message = R._query_message(query)
    if message is None:
        await query.answer()
        return
    await R._drop_flow(state)
    await state.set_state(NewTopicSG.name)
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
    try:
        profile = await R._active_profile(chat_id, assistant_storage)
        topic_id = await topic_storage.create_topic(chat_id, profile.id, name)
        await topic_storage.set_active_topic(chat_id, profile.id, topic_id)
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
        f"✅ Создана новая тема: '{name}'. Контекст обнулен.",
        reply_markup=menu,
    )


async def sg_rename_topic_name(
    message: Message,
    state: FSMContext,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
) -> None:
    """Записывает новое название выбранной темы."""
    name = (message.text or "").strip()
    if not name or len(name) > 100:
        await message.answer(
            "Введите новое название (до 100 символов).\n/cancel — отмена"
        )
        return
    data = await state.get_data()
    topic_id = data.get("flow_topic_id")
    chat_id = message.chat.id
    if not isinstance(topic_id, str) or not topic_id:
        await R._reset_flow(state)
        menu = await R._menu_for(chat_id, assistant_storage, topic_storage)
        await message.answer("Тема для переименования не выбрана.", reply_markup=menu)
        return
    try:
        profile = await R._active_profile(chat_id, assistant_storage)
        await topic_storage.rename_topic(chat_id, profile.id, topic_id, name)
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
    await message.answer(f"Тема переименована в «{name}».")
    await R._open_topics(message, topic_storage, assistant_storage, state)
