"""Исполнение решения роутера: правка, генерация, анализ, уточнение."""

from __future__ import annotations

import os
from typing import Any

import structlog

from app.bot import models as M
from app.bot import runtime as R
from app.bot.photo_router import PhotoDecision, decide
from app.infra.cost_tracker import BudgetExceededError, cost_tracker, set_billing_user
from app.infra.rate_limiter import rate_limiter
from app.infra.security import security
from app.llm.assist_ai import AssistAIError, ImageModelScope
from app.llm.vision import (
    VisionError,
    build_multimodal_content,
    download_telegram_file,
    ensure_within_size_limit,
)
from app.media.media_tools import (
    MediaTurnScope,
    UserMedia,
    reset_user_media,
    set_user_media,
)
from app.storage.assistant_storage import compose_system_prompt
from app.storage.pending_media import PendingMedia, PendingMediaStorage

logger = structlog.get_logger("photo_flow")

_CANCEL = frozenset({"отмена", "стоп", "cancel"})
_ANALYZE_QUESTION = {
    "analyze": "Опиши подробно, что изображено",
    "ocr": "Извлеки весь текст с изображения",
    "compare": "Сравни изображения и опиши различия",
}


def max_clarify_rounds() -> int:
    """Сколько уточнений задать, прежде чем сбросить ожидание."""
    raw = os.getenv("PHOTO_MAX_CLARIFY_ROUNDS", "2").strip()
    try:
        value = int(raw)
    except ValueError:
        return 2
    return value if value > 0 else 2


def confirm_mode() -> str:
    """A — текст персоны, C — шаблон без вызова модели."""
    mode = os.getenv("PHOTO_CONFIRM_MODE", "A").strip().upper()
    return mode if mode in {"A", "C"} else "A"


async def confirm_reply(
    assistant: Any,
    profile: Any,
    user_prompt: str,
    *,
    text_model: str,
    fallback: str,
) -> str:
    """Короткий ответ персоны без фото и без tools. Сбой → шаблон."""
    if confirm_mode() == "C":
        logger.info("confirm_turn", ok=True, mode="C")
        return fallback
    try:
        text = await assistant.chat_completion(
            [
                {
                    "role": "system",
                    "content": compose_system_prompt(profile.system_prompt),
                },
                {"role": "user", "content": user_prompt},
            ],
            model=text_model,
            use_tools=False,
            max_tokens=300,
        )
    except Exception:
        logger.info("confirm_turn", ok=False, mode="A")
        return fallback
    logger.info("confirm_turn", ok=True, mode="A")
    cleaned = text.strip() if isinstance(text, str) else ""
    return cleaned or fallback


async def consume_pending_text(
    message: Any,
    assistant: Any,
    topic_storage: Any,
    assistant_storage: Any,
    state: Any,
    pending: PendingMediaStorage,
    text: str,
) -> bool:
    """True, если текст ушёл роутеру или отменил ожидание фото."""
    try:
        profile = await R._active_profile(message.chat.id, assistant_storage)
    except Exception:
        return False
    media = await pending.get(message.chat.id, profile.id)
    if media is None or not media.items:
        return False
    if text.casefold() in _CANCEL:
        await pending.pop(message.chat.id, profile.id)
        await message.answer("Отменено.")
        await R._reset_flow(state)
        return True
    media.caption = text
    await pending.put(media)
    turn = _turn(message, assistant, topic_storage, assistant_storage, state, pending)
    await route_album(turn, media, charge_text_limit=False)
    return True


async def route_album(
    turn: Any, media: PendingMedia, *, charge_text_limit: bool = True
) -> None:
    """Скачивает фото, спрашивает роутер и исполняет ветку."""
    message = turn.message
    user_id = message.from_user.id if message.from_user else message.chat.id
    set_billing_user(user_id)
    logger.info(
        "album_flushed",
        chat_id=media.chat_id,
        count=len(media.items),
    )
    if charge_text_limit and not await rate_limiter.check_text_limit(user_id):
        retry_after = await rate_limiter.get_retry_after(user_id, "text")
        await message.answer(
            f"⏳ Слишком много запросов. Подождите {int(retry_after) or 60} секунд."
        )
        return
    try:
        profile = await R._active_profile(message.chat.id, turn.assistant_storage)
        topic_id = await R._active_topic(
            message.chat.id, profile.id, turn.topic_storage
        )
    except Exception:
        await R._reply_profile_error(message)
        return
    media.topic_id = topic_id
    loaded = await _download(turn, media)
    if loaded is None:
        return
    images = [{"data": item.data, "mime": item.mime_type} for item in loaded]
    try:
        await cost_tracker.check_budget(
            user_id, cost_tracker.estimate_chat_cost(input_chars=800, output_chars=200)
        )
        decision = await _decision(turn, media, images, topic_id)
    except BudgetExceededError as exc:
        await message.answer(exc.message)
        return
    except AssistAIError as exc:
        await message.answer(exc.message)
        return
    await _execute(turn, media, profile, loaded, decision, topic_id)


async def _decision(
    turn: Any,
    media: PendingMedia,
    images: list[dict[str, Any]],
    topic_id: str,
) -> PhotoDecision:
    caption = media.caption.strip()
    if os.getenv("PHOTO_ROUTER_FAST_PATH", "0").strip() == "1":
        from app.bot.photo_routing import resolve_roles

        roles = resolve_roles(caption, len(images), base_override=media.base_override)
        if roles is not None:
            base, refs = roles
            return PhotoDecision(
                intent="edit",
                base_index=base,
                reference_indexes=refs,
                edit_type="generic",
                instruction_en=caption,
                instruction_ru=caption,
                confidence=1.0,
            )
    context = await _text_context(
        turn.topic_storage, media.chat_id, media.assistant_id, topic_id
    )
    return await decide(turn.assistant, images, caption, context)


async def _execute(
    turn: Any,
    media: PendingMedia,
    profile: Any,
    loaded: list[UserMedia],
    decision: PhotoDecision,
    topic_id: str,
) -> None:
    message = turn.message
    if decision.intent == "unclear":
        await _unclear(turn, media, decision)
        return
    if decision.intent == "chat":
        await turn.pending.pop(media.chat_id, media.assistant_id)
        text = media.caption.strip() or (decision.instruction_ru or "")
        if text:
            await R._process_user_message(
                message,
                turn.assistant,
                turn.topic_storage,
                turn.assistant_storage,
                turn.state,
                user_text=security.sanitize_prompt(text),
                input_kind="text",
            )
        return
    if decision.intent in {"analyze", "ocr", "compare"}:
        await _analyze(turn, media, loaded, decision)
        await turn.pending.pop(media.chat_id, media.assistant_id)
        await R._reset_flow(turn.state)
        return
    if decision.intent == "generate":
        await _generate(turn, media, profile, decision, topic_id)
        return
    await _edit(turn, media, profile, loaded, decision, topic_id)


async def _unclear(turn: Any, media: PendingMedia, decision: PhotoDecision) -> None:
    if media.clarify_rounds >= max_clarify_rounds():
        await turn.pending.pop(media.chat_id, media.assistant_id)
        await turn.message.answer("Опишите, что сделать, одним сообщением.")
        await R._reset_flow(turn.state)
        return
    question = (decision.question_ru or "").strip() or (
        "Опишите, что сделать с фото одним сообщением."
    )
    await turn.message.answer(question)
    await turn.pending.bump_round(media.chat_id, media.assistant_id, question)


async def _analyze(
    turn: Any,
    media: PendingMedia,
    loaded: list[UserMedia],
    decision: PhotoDecision,
) -> None:
    question = (media.caption or "").strip() or _ANALYZE_QUESTION[decision.intent]
    from app.llm.vision import build_image_content

    parts = [build_image_content(item.data, item.mime_type) for item in loaded]
    content = build_multimodal_content(question, parts)
    placeholder = f"[Альбом: {len(loaded)} фото]"
    await R._process_vision_message(
        turn.message,
        turn.assistant,
        turn.topic_storage,
        turn.assistant_storage,
        turn.state,
        multimodal_content=content,
        history_placeholder=placeholder,
        user_media=loaded,
        status_text="Смотрю фото…",
    )


async def _edit(
    turn: Any,
    media: PendingMedia,
    profile: Any,
    loaded: list[UserMedia],
    decision: PhotoDecision,
    topic_id: str,
) -> None:
    message = turn.message
    user_id = message.from_user.id if message.from_user else message.chat.id
    if not await rate_limiter.check_image_limit(user_id):
        retry_after = await rate_limiter.get_retry_after(user_id, "image")
        await message.answer(
            f"⏳ Слишком много запросов картинок. Подождите {int(retry_after) or 60} секунд."
        )
        return
    instruction = (decision.instruction_en or media.caption or "").strip()
    if not security.validate_prompt_length(instruction):
        await message.answer("⚠️ Запрос слишком длинный.")
        return
    base = decision.base_index if decision.base_index is not None else 0
    refs = list(decision.reference_indexes)
    ordered = [loaded[base], *[loaded[index] for index in refs]]
    files = [(item.data, item.filename, item.mime_type) for item in ordered]
    labels = [str(base + 1), *[str(index + 1) for index in refs]]
    model = M._image_model_of(profile, turn.assistant, chat_id=media.chat_id)
    prompt = security.sanitize_prompt(instruction)
    if refs:
        prompt = f"{prompt}\nImage 1 is the base (photo {base + 1}). " + " ".join(
            f"Image {offset + 2} is reference photo {index + 1}."
            for offset, index in enumerate(refs)
        )
    logger.info(
        "edit_image_call",
        base_index=base,
        reference_indexes=refs,
        edit_type=decision.edit_type,
        model=model,
    )
    status = await _status(turn, "Правлю изображение…")
    token = set_user_media(ordered)
    try:
        try:
            await cost_tracker.check_budget(
                user_id, cost_tracker.estimate_image_cost(1)
            )
        except BudgetExceededError as exc:
            await message.answer(exc.message)
            return
        import app.infra.job_queue as job_queue

        with ImageModelScope(model, M._stored_image_size(profile)), MediaTurnScope():
            try:
                result = await job_queue.edit_images(
                    turn.assistant,
                    files,
                    prompt,
                    model=model,
                    labels=labels,
                    size=M._stored_image_size(profile),
                )
            except AssistAIError as exc:
                if "несколько изображений" in exc.message:
                    logger.info(
                        "multi_image_unsupported",
                        model=model,
                        count=len(files),
                    )
                await message.answer(exc.message)
                return
        for index, image in enumerate(result):
            await R._send_generated_image(message, image, index)
        logger.info("edit_image_done", count=len(result), base_index=base)
        fallback = f"Готово. Основа — фото {base + 1}."
        shown_refs = ", ".join(str(index + 1) for index in refs) or "нет"
        user_prompt = (
            f"Пользователь прислал {len(loaded)} фото и попросил: "
            f"{decision.instruction_ru or media.caption}. "
            f"Правка выполнена: основа — фото {base + 1}, референсы — {shown_refs}. "
            "Изображение уже отправлено в Telegram. Кратко подтверди в своём стиле, "
            "не описывай фото и не вызывай инструменты."
        )
        text_model = M._text_model_of(profile, turn.assistant, chat_id=media.chat_id)
        try:
            await cost_tracker.check_budget(
                user_id,
                cost_tracker.estimate_chat_cost(input_chars=400, output_chars=300),
            )
            reply = await confirm_reply(
                turn.assistant,
                profile,
                user_prompt,
                text_model=text_model,
                fallback=fallback,
            )
        except BudgetExceededError as exc:
            await message.answer(exc.message)
            return
        await message.answer(reply)
        await _remember_pair(
            turn,
            topic_id,
            f"[Альбом: {len(loaded)} фото, основа — фото {base + 1}] "
            f"{decision.instruction_ru or ''}".strip(),
            reply,
        )
    finally:
        reset_user_media(token)
        await _drop_status(turn, status)
        await turn.pending.pop(media.chat_id, media.assistant_id)
        await R._reset_flow(turn.state)


async def _generate(
    turn: Any,
    media: PendingMedia,
    profile: Any,
    decision: PhotoDecision,
    topic_id: str,
) -> None:
    message = turn.message
    user_id = message.from_user.id if message.from_user else message.chat.id
    if not await rate_limiter.check_image_limit(user_id):
        retry_after = await rate_limiter.get_retry_after(user_id, "image")
        await message.answer(
            f"⏳ Слишком много запросов картинок. Подождите {int(retry_after) or 60} секунд."
        )
        return
    instruction = (decision.instruction_en or media.caption or "").strip()
    if not security.validate_prompt_length(instruction):
        await message.answer("⚠️ Запрос слишком длинный.")
        return
    model = M._image_model_of(profile, turn.assistant, chat_id=media.chat_id)
    size = M._stored_image_size(profile)
    status = await _status(turn, "Рисую изображение…")
    try:
        try:
            await cost_tracker.check_budget(
                user_id, cost_tracker.estimate_image_cost(1)
            )
        except BudgetExceededError as exc:
            await message.answer(exc.message)
            return
        import app.infra.job_queue as job_queue

        with ImageModelScope(model, size), MediaTurnScope():
            result = await job_queue.generate_image(
                turn.assistant,
                security.sanitize_prompt(instruction),
                model=model,
                size=size,
            )
        for index, image in enumerate(result):
            await R._send_generated_image(message, image, index)
        fallback = "Готово. Это новое изображение по вашему фото."
        user_prompt = (
            f"Пользователь прислал {len(media.items)} фото и попросил: "
            f"{decision.instruction_ru or media.caption}. "
            "Создано новое изображение по референсу, оно уже в Telegram. "
            "Кратко подтверди в своём стиле, не описывай фото и не вызывай инструменты."
        )
        text_model = M._text_model_of(profile, turn.assistant, chat_id=media.chat_id)
        try:
            await cost_tracker.check_budget(
                user_id,
                cost_tracker.estimate_chat_cost(input_chars=400, output_chars=300),
            )
            reply = await confirm_reply(
                turn.assistant,
                profile,
                user_prompt,
                text_model=text_model,
                fallback=fallback,
            )
        except BudgetExceededError as exc:
            await message.answer(exc.message)
            return
        await message.answer(reply)
        await _remember_pair(
            turn,
            topic_id,
            f"[Альбом: {len(media.items)} фото] {decision.instruction_ru or ''}".strip(),
            reply,
        )
    except AssistAIError as exc:
        await message.answer(exc.message)
    finally:
        await _drop_status(turn, status)
        await turn.pending.pop(media.chat_id, media.assistant_id)
        await R._reset_flow(turn.state)


async def _remember_pair(turn: Any, topic_id: str, user_text: str, answer: str) -> None:
    storage = turn.topic_storage
    chat_id = turn.message.chat.id
    try:
        profile = await R._active_profile(chat_id, turn.assistant_storage)
    except Exception:
        return
    limit = storage.max_length
    try:
        await storage.append_message(
            chat_id,
            profile.id,
            topic_id,
            {"role": "user", "content": user_text},
            limit,
        )
        await storage.append_message(
            chat_id,
            profile.id,
            topic_id,
            {"role": "assistant", "content": answer},
            limit,
        )
    except Exception:
        logger.info("photo_history_skipped", chat_id=chat_id)


async def _text_context(
    storage: Any, chat_id: int, assistant_id: str, topic_id: str
) -> list[dict[str, Any]]:
    try:
        history = await storage.get_history(chat_id, assistant_id, topic_id)
    except Exception:
        return []
    texts: list[dict[str, Any]] = []
    if not isinstance(history, list):
        return texts
    for item in history:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if item.get("role") in {"user", "assistant"} and isinstance(content, str):
            texts.append({"role": item["role"], "content": content})
    return texts[-3:]


async def _download(turn: Any, media: PendingMedia) -> list[UserMedia] | None:
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
    logger.info("photo_received", chat_id=media.chat_id, count=len(loaded))
    return loaded


async def _status(turn: Any, text: str) -> Any:
    keep = R._KEEP_OUTGOING.set(True)
    status = await turn.message.answer(text)
    R._release_outgoing(keep)
    await R._append_chrome(turn.state, [status.message_id])
    return status


async def _drop_status(turn: Any, status: Any) -> None:
    if status is None:
        return
    await R._delete_bot_message(status)


def _turn(
    message: Any,
    assistant: Any,
    topic_storage: Any,
    assistant_storage: Any,
    state: Any,
    pending: PendingMediaStorage,
) -> Any:
    from app.bot.handlers.photo import PhotoTurn

    return PhotoTurn(
        message=message,
        assistant=assistant,
        topic_storage=topic_storage,
        assistant_storage=assistant_storage,
        state=state,
        pending=pending,
    )
