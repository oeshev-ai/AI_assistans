"""Маршрут фото: намерение, роли, экран и вызов edit_image."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    UserMedia,
)
from app.bot.photo_routing import _photo_intent, resolve_roles
from app.bot.states import PhotoSG
from app.infra.rate_limiter import rate_limiter
from app.infra.security import security
from app.llm.vision import (
    VisionError,
    build_image_content,
    build_multimodal_content,
    download_telegram_file,
    ensure_within_size_limit,
)
from app.media.media_tools import (
    mark_image_limit_checked,
    reset_base_override,
    reset_image_limit_checked,
    set_base_override,
    take_last_edit_base,
)
from app.storage.pending_media import (
    PendingMedia,
    PendingMediaItem,
    PendingMediaStorage,
)

logger = structlog.get_logger("photo")

_EDIT_TOOL_CHOICE = {"type": "function", "function": {"name": "edit_image"}}


@dataclass
class PhotoTurn:
    """Сервисы, которые отложенная задача альбома не получит из DI."""

    message: Message
    assistant: AssistAIClient
    topic_storage: TopicHistoryStorage
    assistant_storage: AssistantStorage
    state: FSMContext
    pending: PendingMediaStorage


def _history_placeholder(count: int, base: int | None) -> str:
    if base is None:
        return f"[Альбом: {count} фото]"
    return f"[Альбом: {count} фото, основа — фото {base + 1}]"


def _edit_contract(instruction: str, base: int, refs: list[int]) -> str:
    return (
        "Пользователь просит изменить фото. "
        f"Основа base_index={base} (фото {base + 1}). "
        f"Референсы reference_indexes={refs}. "
        "Вызови edit_image именно с этими индексами. "
        f"instruction: {instruction}"
    )


def _vision_edit_text(instruction: str, count: int) -> str:
    last = max(count - 1, 0)
    return (
        f"Во вложении {count} фото. Индексы идут с 0 по {last} "
        "(фото 1 — индекс 0) в порядке вложений, это не роли. "
        f"Пользователь просит правку: {instruction}. "
        "Вызови edit_image и укажи base_index и reference_indexes. "
        "Не описывай фото текстом."
    )


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
    intent = _photo_intent(built.caption)
    logger.info(
        "photo_intent",
        chat_id=message.chat.id,
        intent=intent,
        count=len(media.items),
    )
    if intent == "ask":
        await _show_action(message, turn.state, media)
        return
    if intent == "analyze":
        await _run_analyze(turn, media, built.caption or "Опиши, что на фото.")
        return
    await _run_edit(turn, media, built.caption)


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


async def _download(turn: PhotoTurn, media: PendingMedia) -> list[UserMedia] | None:
    bot = turn.message.bot
    if bot is None:
        await turn.message.answer("Не удалось загрузить файл")
        return None
    loaded: list[UserMedia] = []
    try:
        for item in media.items:
            data = await download_telegram_file(bot, item.file_id)
            ensure_within_size_limit(data, label="Фото")
            loaded.append(
                UserMedia(
                    kind="image",
                    filename=item.filename,
                    mime_type=item.mime_type,
                    data=data,
                )
            )
    except VisionError as exc:
        await turn.message.answer(exc.message)
        return None
    return loaded


def _parts(loaded: list[UserMedia]) -> list[dict[str, Any]]:
    return [build_image_content(item.data, item.mime_type) for item in loaded]


async def _run_analyze(turn: PhotoTurn, media: PendingMedia, question: str) -> None:
    loaded = await _download(turn, media)
    if loaded is None:
        return
    placeholder = _history_placeholder(len(loaded), None)
    await R._process_vision_message(
        turn.message,
        turn.assistant,
        turn.topic_storage,
        turn.assistant_storage,
        turn.state,
        multimodal_content=build_multimodal_content(question, _parts(loaded)),
        history_placeholder=placeholder,
        user_media=loaded,
        status_text="Смотрю фото…",
    )
    await turn.pending.clear(media.chat_id, media.assistant_id)
    await R._reset_flow(turn.state)


async def _run_edit(turn: PhotoTurn, media: PendingMedia, instruction: str) -> None:
    user_id = (
        turn.message.from_user.id if turn.message.from_user else turn.message.chat.id
    )
    if not await rate_limiter.check_image_limit(user_id):
        retry_after = await rate_limiter.get_retry_after(user_id, "image")
        await turn.message.answer(
            f"⏳ Слишком много запросов картинок. Подождите {int(retry_after) or 60} секунд."
        )
        return
    cleaned = security.sanitize_prompt(instruction.strip())
    roles = resolve_roles(cleaned, len(media.items), base_override=media.base_override)
    logger.info(
        "roles_resolved",
        chat_id=media.chat_id,
        count=len(media.items),
        explicit=roles is not None,
        base_index=None if roles is None else roles[0],
        reference_indexes=None if roles is None else roles[1],
    )
    loaded = await _download(turn, media)
    if loaded is None:
        return
    limit_token = mark_image_limit_checked()
    base_token = set_base_override(None if roles is None else roles[0])
    try:
        if roles is None:
            edited = await _edit_via_vision(turn, media, loaded, cleaned)
        else:
            edited = await _edit_with_roles(turn, media, loaded, cleaned, roles)
    finally:
        reset_base_override(base_token)
        reset_image_limit_checked(limit_token)
    if edited:
        base = take_last_edit_base()
        if base is not None:
            await turn.message.answer(f"Основа: фото {base + 1}")
            await turn.pending.clear(media.chat_id, media.assistant_id)
            await R._reset_flow(turn.state)
        return
    await _ask_base_once(turn, media)


async def _edit_with_roles(
    turn: PhotoTurn,
    media: PendingMedia,
    loaded: list[UserMedia],
    instruction: str,
    roles: tuple[int, list[int]],
) -> bool:
    base, refs = roles
    placeholder = _history_placeholder(len(loaded), base)
    return await R._process_user_message(
        turn.message,
        turn.assistant,
        turn.topic_storage,
        turn.assistant_storage,
        turn.state,
        user_text=_edit_contract(instruction, base, refs),
        history_text=placeholder,
        user_media=loaded,
        tool_choice=_EDIT_TOOL_CHOICE,
        suppress_text_without_tools=True,
        status=None,
    )


async def _edit_via_vision(
    turn: PhotoTurn,
    media: PendingMedia,
    loaded: list[UserMedia],
    instruction: str,
) -> bool:
    placeholder = _history_placeholder(len(loaded), None)
    content = build_multimodal_content(
        _vision_edit_text(instruction, len(loaded)),
        _parts(loaded),
    )
    return await R._process_vision_message(
        turn.message,
        turn.assistant,
        turn.topic_storage,
        turn.assistant_storage,
        turn.state,
        multimodal_content=content,
        history_placeholder=placeholder,
        user_media=loaded,
        status_text="Готовлю правку…",
        tool_choice=_EDIT_TOOL_CHOICE,
        suppress_text_without_tools=True,
    )


async def _ask_base_once(turn: PhotoTurn, media: PendingMedia) -> None:
    if media.role_retry_used or len(media.items) < 2:
        await turn.message.answer(
            "Не удалось понять, какое фото основа. "
            "Напишите, например: «основа — фото 2»."
        )
        return
    await turn.pending.mark_role_retry(media.chat_id, media.assistant_id)
    await turn.state.set_state(PhotoSG.action)
    text, markup = screens._photo_base_pick_view(len(media.items))
    await R._upsert_screen(turn.message, turn.state, _SCREEN_PHOTO, text, markup)


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
