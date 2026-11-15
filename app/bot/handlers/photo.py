"""Маршрут фото: намерение, роли, экран и вызов edit_image."""

from __future__ import annotations

from dataclasses import dataclass

import structlog
from aiogram.types import Message

from app.bot import runtime as R
from app.bot import screens
from app.bot.album import AlbumBuild, IncomingPhoto
from app.bot.constants import (
    _SCREEN_PHOTO,
    AssistAIClient,
    AssistantStorage,
    CallbackQuery,
    FSMContext,
    TopicHistoryStorage,
)
from app.bot.states import PhotoSG
from app.infra.security import security
from app.storage.pending_media import (
    PendingMedia,
    PendingMediaItem,
    PendingMediaStorage,
)

logger = structlog.get_logger("photo")


@dataclass
class PhotoTurn:
    """Сервисы, которые отложенная задача альбома не получит из DI."""

    message: Message
    assistant: AssistAIClient
    topic_storage: TopicHistoryStorage
    assistant_storage: AssistantStorage
    state: FSMContext
    pending: PendingMediaStorage


async def deliver_album(built: AlbumBuild, turn: PhotoTurn) -> None:
    """Дальше по намерению: экран, описание или правка."""
    message = turn.message
    if built.error:
        await message.answer(built.error)
        return
    if not built.photos:
        return
    if built.notice:
        await message.answer(built.notice)
    try:
        profile = await R._active_profile(message.chat.id, turn.assistant_storage)
    except Exception:
        await R._reply_profile_error(message)
        return
    media = PendingMedia(
        chat_id=message.chat.id,
        assistant_id=profile.id,
        items=[
            PendingMediaItem(
                file_id=item.file_id,
                message_id=item.message_id,
                file_size=item.size,
                filename=item.filename,
                mime_type=item.mime,
            )
            for item in built.photos
        ],
        caption=built.caption,
    )
    await turn.pending.put(media)
    from app.bot.photo_flow import route_album

    await route_album(turn, media)


async def _show_action(
    message: Message, state: FSMContext, media: PendingMedia
) -> None:
    await state.set_state(PhotoSG.action)
    await state.update_data(flow_photo_id=media.assistant_id)
    text, markup = screens._photo_action_view(len(media.items), media.base_override)
    await R._upsert_screen(message, state, _SCREEN_PHOTO, text, markup)


async def _load(
    turn: PhotoTurn,
) -> tuple[PendingMedia, str] | None:
    try:
        profile = await R._active_profile(turn.message.chat.id, turn.assistant_storage)
    except Exception:
        await R._reply_profile_error(turn.message)
        return None
    media = await turn.pending.get(turn.message.chat.id, profile.id)
    if media is None or not media.items:
        await turn.message.answer("Фото устарели. Пришлите их ещё раз.")
        await R._reset_flow(turn.state)
        return None
    return media, profile.id


async def _run_analyze(turn: PhotoTurn, media: PendingMedia, question: str) -> None:
    media.caption = question
    await turn.pending.put(media)
    from app.bot.photo_flow import route_album

    await route_album(turn, media, charge_text_limit=False)


async def _run_edit(turn: PhotoTurn, media: PendingMedia, instruction: str) -> None:
    media.caption = instruction
    await turn.pending.put(media)
    from app.bot.photo_flow import route_album

    await route_album(turn, media, charge_text_limit=False)


def _turn_from(
    message: Message,
    assistant: AssistAIClient,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    state: FSMContext,
    pending: PendingMediaStorage,
) -> PhotoTurn:
    return PhotoTurn(
        message=message,
        assistant=assistant,
        topic_storage=topic_storage,
        assistant_storage=assistant_storage,
        state=state,
        pending=pending,
    )


async def on_photo_edit(
    callback: CallbackQuery,
    assistant: AssistAIClient,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    state: FSMContext,
    pending_media: PendingMediaStorage,
) -> None:
    await callback.answer()
    message = callback.message
    if not isinstance(message, Message):
        return
    await state.set_state(PhotoSG.edit_instruction)
    await message.answer("Что изменить на фото?")


async def on_photo_analyze(
    callback: CallbackQuery,
    assistant: AssistAIClient,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    state: FSMContext,
    pending_media: PendingMediaStorage,
) -> None:
    await callback.answer()
    message = callback.message
    if not isinstance(message, Message):
        return
    turn = _turn_from(
        message, assistant, topic_storage, assistant_storage, state, pending_media
    )
    loaded = await _load(turn)
    if loaded is None:
        return
    media, _assistant_id = loaded
    question = media.caption.strip() or "Опиши, что на фото."
    await _run_analyze(turn, media, question)


async def on_photo_ask(
    callback: CallbackQuery,
    state: FSMContext,
) -> None:
    await callback.answer()
    message = callback.message
    if not isinstance(message, Message):
        return
    await state.set_state(PhotoSG.question)
    await message.answer("Какой вопрос по фото?")


async def on_photo_cancel(
    callback: CallbackQuery,
    assistant_storage: AssistantStorage,
    state: FSMContext,
    pending_media: PendingMediaStorage,
) -> None:
    await callback.answer()
    message = callback.message
    if isinstance(message, Message):
        try:
            profile = await R._active_profile(message.chat.id, assistant_storage)
            await pending_media.clear(message.chat.id, profile.id)
        except Exception:
            logger.info("photo_cancel_without_profile")
        await message.answer("Отменено.")
    await R._reset_flow(state)


async def on_photo_base_next(
    callback: CallbackQuery,
    assistant_storage: AssistantStorage,
    state: FSMContext,
    pending_media: PendingMediaStorage,
) -> None:
    await callback.answer()
    message = callback.message
    if not isinstance(message, Message):
        return
    try:
        profile = await R._active_profile(message.chat.id, assistant_storage)
    except Exception:
        await R._reply_profile_error(message)
        return
    media = await pending_media.get(message.chat.id, profile.id)
    if media is None or not media.items:
        await message.answer("Фото устарели. Пришлите их ещё раз.")
        return
    count = len(media.items)
    current = media.base_override
    nxt: int | None = 0 if current is None else current + 1
    if nxt is not None and nxt >= count:
        nxt = None
    updated = await pending_media.set_base_override(message.chat.id, profile.id, nxt)
    shown = updated or media
    text, markup = screens._photo_action_view(count, shown.base_override)
    await R._upsert_screen(message, state, _SCREEN_PHOTO, text, markup)


async def on_photo_base_set(
    callback: CallbackQuery,
    assistant: AssistAIClient,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    state: FSMContext,
    pending_media: PendingMediaStorage,
) -> None:
    await callback.answer()
    message = callback.message
    if not isinstance(message, Message) or not callback.data:
        return
    raw = callback.data.removeprefix("photo:base_set:")
    try:
        index = int(raw)
    except ValueError:
        return
    turn = _turn_from(
        message, assistant, topic_storage, assistant_storage, state, pending_media
    )
    loaded = await _load(turn)
    if loaded is None:
        return
    _media, assistant_id = loaded
    updated = await pending_media.set_base_override(
        message.chat.id, assistant_id, index
    )
    if updated is None:
        await message.answer("Фото устарели. Пришлите их ещё раз.")
        return
    instruction = updated.caption.strip() or "Измени фото, основа выбрана вручную."
    await _run_edit(turn, updated, instruction)


async def sg_photo_edit_instruction(
    message: Message,
    assistant: AssistAIClient,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    state: FSMContext,
    pending_media: PendingMediaStorage,
) -> None:
    text = (message.text or "").strip()
    if not text:
        return
    turn = _turn_from(
        message, assistant, topic_storage, assistant_storage, state, pending_media
    )
    loaded = await _load(turn)
    if loaded is None:
        return
    media, _assistant_id = loaded
    media.caption = text
    await pending_media.put(media)
    await _run_edit(turn, media, text)


async def sg_photo_question(
    message: Message,
    assistant: AssistAIClient,
    topic_storage: TopicHistoryStorage,
    assistant_storage: AssistantStorage,
    state: FSMContext,
    pending_media: PendingMediaStorage,
) -> None:
    text = (message.text or "").strip()
    if not text:
        return
    turn = _turn_from(
        message, assistant, topic_storage, assistant_storage, state, pending_media
    )
    loaded = await _load(turn)
    if loaded is None:
        return
    media, _assistant_id = loaded
    await _run_analyze(turn, media, security.sanitize_prompt(text))


def incoming_from_message(message: Message) -> IncomingPhoto | None:
    if not message.photo:
        return None
    largest = message.photo[-1]
    return IncomingPhoto(
        message_id=message.message_id,
        file_id=largest.file_id,
        size=largest.file_size or 0,
        caption=(message.caption or "").strip(),
    )
