"""Исполнение решения роутера: правка, генерация, анализ, уточнение."""

from __future__ import annotations

from typing import Any

import structlog

from app.bot import models as M
from app.bot import runtime as R
from app.bot.photo_router import PhotoDecision, asks_for_video, decide
from app.infra.cost_tracker import (
    BudgetExceededError,
    cost_tracker,
    set_billing_user,
)
from app.infra.rate_limiter import rate_limiter
from app.infra.security import security
from app.llm.errors import AssistAIError
from app.llm.image_ops import (
    ImageModelScope,
    image_model_family,
    supports_multi_image,
)
from app.llm.video_settings import video_settings_from
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
from app.storage.pending_media import (
    PendingMedia,
    PendingMediaItem,
    PendingMediaStorage,
)

logger = structlog.get_logger("photo_flow")


def _edit_size_for_model(
    profile_size: str, generate_model: str, edit_model: str
) -> str | None:
    """Размер персоны уходит в правку только если семьи совпадают.

    У «Помощника» генерация на FLUX (``1K``), а правка на GPT Image.
    ``1K`` в форму OpenAI не подставляется: размер считается по исходнику.
    """
    if image_model_family(edit_model) == image_model_family(generate_model):
        return profile_size
    return None


def _generate_with_refs_model(generate_model: str, edit_model: str) -> str:
    """Model for intent=generate when reference photos are present.

    ``generate_image`` cannot take input images, so the call stays on
    ``edit_images``. Prefer the persona generate model when it supports
    multi-image edits; otherwise keep the dedicated edit model (e.g. FLUX →
    GPT Image).
    """
    if supports_multi_image(generate_model):
        return generate_model
    return edit_model


_CANCEL = frozenset({"отмена", "стоп", "cancel"})
_ANALYZE_QUESTION = {
    "analyze": "Опиши подробно, что изображено",
    "ocr": "Извлеки весь текст с изображения",
    "compare": "Сравни изображения и опиши различия",
}


def max_clarify_rounds() -> int:
    """Сколько уточнений задать, прежде чем сбросить ожидание."""
    from app.infra.settings import get_settings

    return get_settings().photo_max_clarify_rounds


def confirm_mode() -> str:
    """A — текст персоны, C — шаблон без вызова модели."""
    from app.infra.settings import get_settings

    return get_settings().photo_confirm_mode


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
            # Gemini не даёт выключить рассуждение (HTTP 400), а оно съедает
            # общий max_tokens и обрывает текст. Ограничиваем только его.
            reasoning={"max_tokens": 64},
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
    user_id = message.from_user.id if message.from_user else message.chat.id
    # Claim immediately so concurrent texts cannot both start paid work.
    media = await pending.pop(message.chat.id, profile.id, user_id)
    if media is None or not media.items:
        return False
    if text.casefold() in _CANCEL:
        await message.answer("Отменено.")
        await R._reset_flow(state)
        return True
    media.caption = text
    turn = _turn(message, assistant, topic_storage, assistant_storage, state, pending)
    await route_album(turn, media, charge_text_limit=False, pending_claimed=True)
    return True


async def route_album(
    turn: Any,
    media: PendingMedia,
    *,
    charge_text_limit: bool = True,
    pending_claimed: bool = False,
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
        # deliver_album уже сделал put — слот нельзя оставлять, иначе следующий
        # свободный текст уйдёт как подпись к этому альбому.
        await _release_album(turn, media, None)
        return
    try:
        profile = await R._active_profile(message.chat.id, turn.assistant_storage)
        topic_id = await R._active_topic(
            message.chat.id, profile.id, turn.topic_storage
        )
    except Exception:
        await R._reply_profile_error(message)
        if pending_claimed:
            await _restore_album(turn, media)
        return
    media.topic_id = topic_id
    loaded = await _download(turn, media)
    if loaded is None:
        if pending_claimed:
            await _restore_album(turn, media)
        return
    images = [{"data": item.data, "mime": item.mime_type} for item in loaded]
    try:
        await cost_tracker.check_budget(
            user_id, cost_tracker.estimate_chat_cost(input_chars=800, output_chars=200)
        )
        decision = await _decision(turn, media, images, topic_id)
    except BudgetExceededError as exc:
        if pending_claimed:
            await _restore_album(turn, media)
        await message.answer(exc.message)
        return
    except AssistAIError as exc:
        if pending_claimed:
            await _restore_album(turn, media)
        await message.answer(exc.message)
        return
    finally:
        await cost_tracker.release_reservation(user_id)
    await _execute(
        turn,
        media,
        profile,
        loaded,
        decision,
        topic_id,
        pending_claimed=pending_claimed,
    )


def result_file_id(sent: Any) -> str | None:
    """``file_id`` самого крупного фото или документа отправленного результата."""
    photos = getattr(sent, "photo", None)
    if photos:
        file_id = getattr(photos[-1], "file_id", None)
        if isinstance(file_id, str) and file_id:
            return file_id
    document = getattr(sent, "document", None)
    file_id = getattr(document, "file_id", None) if document is not None else None
    if isinstance(file_id, str) and file_id:
        return file_id
    return None


def caption_from_resend(context: list[dict[str, Any]]) -> str | None:
    """Прошлая просьба, если ассистент просил прислать фото снова.

    ``[Альбом: …]`` и обычный диалог без такой просьбы не становятся подписью.
    """
    last_user = ""
    last_assistant = ""
    for item in context:
        role = item.get("role")
        content = item.get("content")
        if not isinstance(content, str):
            continue
        if role == "user":
            last_user = content.strip()
        elif role == "assistant":
            last_assistant = content.strip()
    if not last_user or last_user.startswith("[Альбом") or not last_assistant:
        return None
    folded = last_assistant.casefold()
    asks = any(
        hint in folded for hint in ("недоступ", "пришлите", "прислать", "пришли")
    )
    mentions = "фото" in folded or "картинк" in folded
    if asks and mentions:
        return last_user
    return None


def followup_media(media: PendingMedia, file_id: str) -> PendingMedia:
    """Одно фото результата с пустой подписью — база для следующей фразы."""
    return PendingMedia(
        chat_id=media.chat_id,
        user_id=media.user_id,
        assistant_id=media.assistant_id,
        items=[
            PendingMediaItem(
                file_id=file_id,
                message_id=0,
                filename="result.png",
                mime_type="image/png",
            )
        ],
        caption="",
        topic_id=media.topic_id,
    )


def _album_identity(media: PendingMedia) -> tuple[tuple[str, int], ...]:
    """Пара file_id и message_id: чужой альбом в том же слоте не трогаем."""
    return tuple((item.file_id, item.message_id) for item in media.items)


async def _restore_album(turn: Any, media: PendingMedia) -> None:
    """Кладёт альбом обратно, если слот пуст или там всё ещё он сам."""
    try:
        current = await turn.pending.get(
            media.chat_id, media.assistant_id, media.user_id
        )
    except Exception:
        logger.info("pending_restore_skipped", chat_id=media.chat_id)
        return
    if current is not None and _album_identity(current) != _album_identity(media):
        return
    await turn.pending.put(media)


async def _release_album(
    turn: Any, media: PendingMedia, kept_file_id: str | None
) -> None:
    """Снимает свой альбом. Альбом, пришедший во время хода, остаётся."""
    try:
        current = await turn.pending.get(
            media.chat_id, media.assistant_id, media.user_id
        )
    except Exception:
        logger.info("pending_release_skipped", chat_id=media.chat_id)
        return
    if current is not None and _album_identity(current) != _album_identity(media):
        return
    await turn.pending.pop(media.chat_id, media.assistant_id, media.user_id)
    if kept_file_id:
        await _keep_result(turn, media, kept_file_id)


async def _keep_result(turn: Any, media: PendingMedia, file_id: str) -> None:
    await turn.pending.put(followup_media(media, file_id))


async def _settle_album(
    turn: Any, media: PendingMedia, kept_file_id: str | None
) -> None:
    """Успех оставляет файл результата. Иначе в слот возвращается исходный альбом."""
    if kept_file_id:
        await _release_album(turn, media, kept_file_id)
    else:
        await _restore_album(turn, media)
    await R._reset_flow(turn.state)


async def _decision(
    turn: Any,
    media: PendingMedia,
    images: list[dict[str, Any]],
    topic_id: str,
) -> PhotoDecision:
    caption = media.caption.strip()
    context: list[dict[str, Any]] | None = None
    if not caption:
        # Прошлая правка лежит в истории темы как реплика пользователя.
        # Без новой подписи роутер принимает её за команду к этому фото,
        # кроме случая, когда ассистент только что просил прислать снимок снова.
        context = await _text_context(
            turn.topic_storage, media.chat_id, media.assistant_id, topic_id
        )
        reused = caption_from_resend(context)
        if not reused:
            return PhotoDecision(
                intent="unclear",
                question_ru="Опишите, что сделать с фото одним сообщением.",
                confidence=0.0,
            )
        caption = reused
        media.caption = reused
    from app.infra.settings import get_settings

    if get_settings().photo_router_fast_path and not asks_for_video(caption):
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
    if context is None:
        context = await _text_context(
            turn.topic_storage, media.chat_id, media.assistant_id, topic_id
        )
    # Ответ на уже заданный вопрос не отбрасываем из-за низкой уверенности.
    answered = media.clarify_rounds >= 1
    return await decide(
        turn.assistant,
        images,
        caption,
        context,
        min_confidence=0.0 if answered else None,
    )


async def _execute(
    turn: Any,
    media: PendingMedia,
    profile: Any,
    loaded: list[UserMedia],
    decision: PhotoDecision,
    topic_id: str,
    *,
    pending_claimed: bool = False,
) -> None:
    message = turn.message
    if decision.intent == "unclear":
        if pending_claimed:
            await turn.pending.put(media)
        await _unclear(turn, media, decision)
        return
    if not pending_claimed:
        claimed = await turn.pending.pop(
            media.chat_id, media.assistant_id, media.user_id
        )
        if claimed is None:
            logger.info(
                "pending_photo_already_claimed",
                chat_id=media.chat_id,
                user_id=media.user_id,
                intent=decision.intent,
            )
            return
    if decision.intent == "chat":
        text = media.caption.strip() or (decision.instruction_ru or "")
        try:
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
        finally:
            media.caption = ""
            media.clarify_rounds = 0
            media.last_question = None
            await turn.pending.put(media)
        return
    if decision.intent in {"analyze", "ocr", "compare"}:
        try:
            await _analyze(turn, media, loaded, decision)
        finally:
            media.caption = ""
            media.clarify_rounds = 0
            media.last_question = None
            await _restore_album(turn, media)
        await R._reset_flow(turn.state)
        return
    if decision.intent == "video":
        await _generate_video(turn, media, loaded, decision, topic_id, profile)
        return
    if decision.intent == "generate":
        await _generate(turn, media, profile, loaded, decision, topic_id)
        return
    await _edit(turn, media, profile, loaded, decision, topic_id)


async def _unclear(turn: Any, media: PendingMedia, decision: PhotoDecision) -> None:
    if media.clarify_rounds >= max_clarify_rounds():
        await turn.pending.pop(media.chat_id, media.assistant_id, media.user_id)
        await turn.message.answer("Опишите, что сделать, одним сообщением.")
        await R._reset_flow(turn.state)
        return
    question = (decision.question_ru or "").strip() or (
        "Опишите, что сделать с фото одним сообщением."
    )
    await turn.message.answer(question)
    await turn.pending.bump_round(
        media.chat_id, media.assistant_id, media.user_id, question
    )


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
        await _settle_album(turn, media, None)
        return
    instruction = (decision.instruction_en or media.caption or "").strip()
    if not security.validate_prompt_length(instruction):
        await message.answer("⚠️ Запрос слишком длинный.")
        await _settle_album(turn, media, None)
        return
    base = decision.base_index if decision.base_index is not None else 0
    refs = list(decision.reference_indexes)
    ordered = [loaded[base], *[loaded[index] for index in refs]]
    files = [(item.data, item.filename, item.mime_type) for item in ordered]
    labels = [str(base + 1), *[str(index + 1) for index in refs]]
    model = turn.assistant.config.default_image_edit_model
    as_file = M._image_as_file(profile, model)
    # Edit must keep source aspect ratio (IMAGE_EDIT_SIZE_MODE=source).
    # Persona image_size is for generate only — passing it here squashes
    # e.g. 3:2 photos to the persona canvas when generate model is also Flare.
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
        size="source",
    )
    status = await _status(turn, "Правлю изображение…")
    token = set_user_media(ordered)
    kept_file_id: str | None = None
    try:
        import app.infra.job_queue as job_queue

        with ImageModelScope(model, None), MediaTurnScope():
            try:
                # Budget preflight — в AssistAIClient.edit_images.
                result = await job_queue.edit_images(
                    turn.assistant,
                    files,
                    prompt,
                    model=model,
                    labels=labels,
                    size=None,
                )
            except BudgetExceededError as exc:
                await message.answer(exc.message)
                return
            except AssistAIError as exc:
                if "несколько изображений" in exc.message:
                    logger.info(
                        "multi_image_unsupported",
                        model=model,
                        count=len(files),
                    )
                from app.bot.image_retry import image_failure_reply

                await message.answer(
                    await image_failure_reply(
                        media.chat_id, user_id, exc, as_file=as_file
                    )
                )
                return
        try:
            for index, image in enumerate(result):
                sent = await R._send_generated_image(
                    message, image, index, as_file=as_file
                )
                file_id = result_file_id(sent)
                if file_id:
                    kept_file_id = file_id
        except AssistAIError as exc:
            from app.bot.image_retry import image_failure_reply

            await message.answer(
                await image_failure_reply(media.chat_id, user_id, exc, as_file=as_file)
            )
            return
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
        await cost_tracker.release_reservation(user_id)
        reset_user_media(token)
        await _drop_status(turn, status)
        await _settle_album(turn, media, kept_file_id)


def video_references(
    loaded: list[UserMedia], indexes: list[int] | None = None
) -> tuple[list[Any], list[int]]:
    """Кадры ролика и их номера в альбоме (с 1), не больше лимита Seedance.

    Непустой ``indexes`` — выбор роутера в его порядке. Пустой список
    оставляет весь альбом: роутер не назвал подмножество.
    """
    from app.llm.media_api import (
        VIDEO_MAX_REFERENCE_IMAGES,
        VideoReference,
        safe_image_mime,
    )

    positions = list(indexes) if indexes else list(range(len(loaded)))
    refs: list[VideoReference] = []
    numbers: list[int] = []
    for index in positions:
        if index < 0 or index >= len(loaded):
            continue
        item = loaded[index]
        if item.kind != "image" or not item.data:
            continue
        refs.append(
            VideoReference(data=item.data, mime_type=safe_image_mime(item.mime_type))
        )
        numbers.append(index + 1)
        if len(refs) >= VIDEO_MAX_REFERENCE_IMAGES:
            break
    return refs, numbers


def video_prompt(instruction: str, photo_numbers: list[int]) -> str:
    """Промпт ролика: «image N» — порядок вложений, «photo M» — номер в альбоме."""
    text = security.sanitize_prompt(instruction.strip())
    if not photo_numbers:
        return text
    order = ", ".join(
        f"image {offset + 1} is photo {number}"
        for offset, number in enumerate(photo_numbers)
    )
    return f"{text}\nAttached reference images, in order: {order}."


async def _generate_video(
    turn: Any,
    media: PendingMedia,
    loaded: list[UserMedia],
    decision: PhotoDecision,
    topic_id: str,
    profile: Any | None = None,
) -> None:
    """Ролик по присланным фото. Повторный запрос сюда не приходит, пока этот ход жив."""
    from aiogram.types import BufferedInputFile

    message = turn.message
    user_id = message.from_user.id if message.from_user else message.chat.id
    if not await rate_limiter.check_media_limit(user_id):
        retry_after = await rate_limiter.get_retry_after(user_id, "media")
        await message.answer(
            f"Слишком много запросов видео. Подождите {int(retry_after) or 60} секунд."
        )
        await _restore_album(turn, media)
        await R._reset_flow(turn.state)
        return
    instruction = (decision.instruction_en or media.caption or "").strip()
    if not instruction:
        await message.answer("Опишите, что должно происходить в ролике.")
        await _restore_album(turn, media)
        await R._reset_flow(turn.state)
        return
    refs, photo_numbers = video_references(loaded, decision.reference_indexes)
    prompt = video_prompt(instruction, photo_numbers)
    if not prompt.strip():
        await message.answer("Опишите, что должно происходить в ролике.")
        await _restore_album(turn, media)
        await R._reset_flow(turn.state)
        return
    status = await _status(
        turn, "Запускаю ролик по этим фото. Это займёт несколько минут."
    )
    delivered = False
    try:
        settings = video_settings_from(profile)
        video = await turn.assistant.generate_video(
            prompt,
            references=refs or None,
            aspect_ratio=settings.aspect_ratio,
            duration=settings.duration,
            resolution=settings.resolution,
            generate_audio=settings.generate_audio,
        )
        filename = video.filename or "video.mp4"
        payload = BufferedInputFile(video.data, filename=filename)
        try:
            await message.answer_video(video=payload)
        except Exception:
            logger.exception("photo_video_send_failed", chat_id=media.chat_id)
            try:
                await message.answer_document(
                    document=BufferedInputFile(video.data, filename=filename)
                )
            except Exception:
                logger.exception("photo_video_document_failed", chat_id=media.chat_id)
                from app.bot.video_retry import VIDEO_KEEP_TEXT, note_failed_video

                source = getattr(video, "source_url", None)
                saved = False
                if isinstance(source, str) and source:
                    saved = await note_failed_video(
                        media.chat_id, user_id, source, filename
                    )
                if saved:
                    await message.answer(VIDEO_KEEP_TEXT)
                else:
                    await message.answer(
                        "Ролик готов, но отправить его в чат не удалось."
                    )
                return
        delivered = True
        await message.answer("Ролик готов.")
        await _remember_pair(
            turn,
            topic_id,
            f"[Альбом: {len(loaded)} фото] {decision.instruction_ru or media.caption}".strip(),
            "Ролик готов.",
        )
        logger.info(
            "photo_video_done",
            chat_id=media.chat_id,
            references=len(refs),
            photo_numbers=photo_numbers,
        )
    except BudgetExceededError as exc:
        await message.answer(exc.message)
    except AssistAIError as exc:
        from app.bot.video_retry import video_failure_reply

        await message.answer(await video_failure_reply(media.chat_id, user_id, exc))
    except Exception:
        logger.exception("photo_video_failed", chat_id=media.chat_id)
        await message.answer("Ролик запустить не удалось.")
    finally:
        await cost_tracker.release_reservation(user_id)
        await _drop_status(turn, status)
        if delivered:
            await _release_album(turn, media, None)
        else:
            await _restore_album(turn, media)
        await R._reset_flow(turn.state)


async def _generate(
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
        await _settle_album(turn, media, None)
        return
    instruction = (decision.instruction_en or media.caption or "").strip()
    if not security.validate_prompt_length(instruction):
        await message.answer("⚠️ Запрос слишком длинный.")
        await _settle_album(turn, media, None)
        return
    generate_model = M._image_model_of(profile, turn.assistant, chat_id=media.chat_id)
    size = M._stored_image_size(profile)
    # Reference photos must reach the image API; text-only generate drops them.
    edit_model = turn.assistant.config.default_image_edit_model
    files = [(item.data, item.filename, item.mime_type) for item in loaded]
    labels = [str(index + 1) for index in range(len(loaded))]
    prompt = security.sanitize_prompt(instruction)
    if files:
        prompt = f"{prompt}\n" + " ".join(
            f"Image {index + 1} is a reference photo." for index in range(len(files))
        )
    status = await _status(turn, "Рисую изображение…")
    token = set_user_media(loaded)
    kept_file_id: str | None = None
    try:
        import app.infra.job_queue as job_queue

        if files:
            model = _generate_with_refs_model(generate_model, edit_model)
            edit_size = _edit_size_for_model(size, generate_model, model)
        else:
            model = generate_model
            edit_size = None
        as_file = M._image_as_file(profile, model)
        with ImageModelScope(model, size), MediaTurnScope():
            # Budget preflight — в generate_image / edit_images.
            if files:
                result = await job_queue.edit_images(
                    turn.assistant,
                    files,
                    prompt,
                    model=model,
                    labels=labels,
                    size=edit_size,
                )
            else:
                result = await job_queue.generate_image(
                    turn.assistant,
                    prompt,
                    model=generate_model,
                    size=size,
                    background=M._image_background_of(profile, turn.assistant),
                )
        try:
            for index, image in enumerate(result):
                sent = await R._send_generated_image(
                    message, image, index, as_file=as_file
                )
                file_id = result_file_id(sent)
                if file_id:
                    kept_file_id = file_id
        except AssistAIError as exc:
            from app.bot.image_retry import image_failure_reply

            await message.answer(
                await image_failure_reply(media.chat_id, user_id, exc, as_file=as_file)
            )
            return
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
    except BudgetExceededError as exc:
        await message.answer(exc.message)
    except AssistAIError as exc:
        from app.bot.image_retry import image_failure_reply

        await message.answer(
            await image_failure_reply(media.chat_id, user_id, exc, as_file=as_file)
        )
    finally:
        await cost_tracker.release_reservation(user_id)
        reset_user_media(token)
        await _drop_status(turn, status)
        await _settle_album(turn, media, kept_file_id)


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
