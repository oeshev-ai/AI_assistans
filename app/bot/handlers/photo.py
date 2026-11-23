"""Маршрут фото: альбом уходит роутеру, тот решает намерение."""

from __future__ import annotations

from dataclasses import dataclass

from aiogram.types import Message

from app.bot import runtime as R
from app.bot.album import AlbumBuild, IncomingPhoto
from app.bot.constants import (
    AssistAIClient,
    AssistantStorage,
    FSMContext,
    TopicHistoryStorage,
)
from app.storage.pending_media import (
    PendingMedia,
    PendingMediaItem,
    PendingMediaStorage,
)


@dataclass
class PhotoTurn:
    """Сервисы, которые отложенная задача альбома не получит из DI."""

    message: Message
    assistant: AssistAIClient
    topic_storage: TopicHistoryStorage
    assistant_storage: AssistantStorage
    state: FSMContext
    pending: PendingMediaStorage


async def deliver_album(
    built: AlbumBuild, turn: PhotoTurn, *, charge_text_limit: bool = True
) -> None:
    """Кладёт альбом в слот и отдаёт роутеру: правка, картинка, разбор или вопрос."""
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
    user_id = message.from_user.id if message.from_user else message.chat.id
    media = PendingMedia(
        chat_id=message.chat.id,
        user_id=user_id,
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

    await route_album(turn, media, charge_text_limit=charge_text_limit)


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
