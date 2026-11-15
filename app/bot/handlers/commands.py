"""Aiogram handlers."""

from __future__ import annotations

from app.bot import runtime as R
from app.bot.album import get_album_hub
from app.bot.constants import (
    WELCOME,
    Any,
    AssistAIClient,
    AssistantStorage,
    Awaitable,
    BaseMiddleware,
    Callable,
    FSMContext,
    Message,
    TelegramObject,
    TopicHistoryStorage,
)
from app.bot.states import (
    EditAssistantSG,
    NewAssistantSG,
    NewTopicSG,
    RenameTopicSG,
)
from app.storage.pending_media import PendingMediaStorage


async def cmd_start(
    message: Message,
    state: FSMContext,
    assistant_storage: AssistantStorage,
    topic_storage: TopicHistoryStorage,
) -> None:
    """Приветствие, список возможностей и постоянное меню."""
    await R._reset_flow(state)
    menu = await R._menu_for(
        message.chat.id,
        assistant_storage,
        topic_storage,
        ensure_topic=True,
    )
    await message.answer(WELCOME, reply_markup=menu)


async def cmd_cancel(
    message: Message,
    state: FSMContext,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    topic_storage: TopicHistoryStorage,
    pending_media: PendingMediaStorage,
) -> None:
    """Сбрасывает текущий пошаговый ввод. Без шага отвечает, что отменять нечего."""
    had_album = get_album_hub().cancel_chat(message.chat.id)
    had_pending = False
    try:
        profile = await R._active_profile(message.chat.id, assistant_storage)
        had_pending = await pending_media.get(message.chat.id, profile.id) is not None
        await pending_media.clear(message.chat.id, profile.id)
    except Exception:
        had_pending = False
    menu = await R._menu_for(message.chat.id, assistant_storage, topic_storage)
    current = await state.get_state()
    if current is None and not had_album and not had_pending:
        await R._apply_reply_menu(message, menu, state)
        await message.answer("Нечего отменять.")
        return
    if current is None:
        await R._apply_reply_menu(message, menu, state)
        await message.answer("Отменено.")
        return
    return_to_settings = current in {
        EditAssistantSG.name.state,
        EditAssistantSG.prompt.state,
        NewAssistantSG.name.state,
        NewAssistantSG.prompt.state,
        NewAssistantSG.models.state,
    }
    return_to_topics = current in {
        RenameTopicSG.name.state,
        NewTopicSG.name.state,
    }
    await R._reset_flow(state)
    await R._apply_reply_menu(message, menu, state)
    if return_to_settings:
        await R._open_settings(message, assistant, assistant_storage, state)
    elif return_to_topics:
        await R._open_topics(message, topic_storage, assistant_storage, state)


async def cmd_unknown(message: Message) -> None:
    """Отвечает на команду, для которой нет отдельного хендлера."""
    await message.answer("Команды убраны. Пользуйтесь кнопками внизу или /help.")


class _CommandLeavesWizard(BaseMiddleware):
    """Любая команда, кроме /cancel, выходит из мастера до своего хендлера."""

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        if isinstance(event, Message):
            text = event.text or ""
            state = data.get("state")
            if (
                isinstance(state, FSMContext)
                and text.startswith("/")
                and not text.startswith("/cancel")
            ):
                await R._drop_flow(state)
        return await handler(event, data)
