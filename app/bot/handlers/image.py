"""Aiogram handlers."""

from __future__ import annotations

from app.bot import models as M
from app.bot import runtime as R
from app.bot.constants import (
    AVAILABLE_IMAGE_MODELS,
    Any,
    AssistAIClient,
    AssistantStorage,
    CallbackQuery,
    FSMContext,
    Message,
    TopicHistoryStorage,
    image_model_family,
    logger,
    rate_limiter,
    security,
)
from app.bot.states import (
    ImageSG,
)


async def on_image_model(
    query: CallbackQuery, state: FSMContext, assistant: AssistAIClient
) -> None:
    """Меняет модель картинки на экране генерации, не записывая её в персону."""
    raw = (query.data or "").removeprefix("image:model:")
    try:
        selected: str | None = AVAILABLE_IMAGE_MODELS[int(raw)]
    except (ValueError, IndexError):
        await query.answer("Неизвестная модель", show_alert=True)
        return
    data = await state.get_data()
    old_model = (
        M._optional_model(data.get("image_model"))
        or assistant.config.default_image_model
    )
    new_model = selected or assistant.config.default_image_model
    fields: dict[str, Any] = {"image_model": selected}
    if image_model_family(old_model) != image_model_family(new_model):
        fields["image_size"] = M._default_image_size(new_model)
    await state.update_data(fields)
    await R._refresh_image_screen(query, state, assistant)


async def on_image_size(
    query: CallbackQuery, state: FSMContext, assistant: AssistAIClient
) -> None:
    """Меняет размер картинки на экране генерации."""
    size = (query.data or "").removeprefix("image:size:")
    data = await state.get_data()
    selected = M._optional_model(data.get("image_model"))
    model_id = selected or assistant.config.default_image_model
    if size not in M._size_choices(model_id):
        await query.answer("Этот размер не подходит модели", show_alert=True)
        return
    await state.update_data(image_size=size)
    await R._refresh_image_screen(query, state, assistant)


async def on_image_go(query: CallbackQuery, state: FSMContext) -> None:
    """Переходит к вводу описания картинки."""
    message = R._query_message(query)
    if message is None:
        await query.answer()
        return
    await state.set_state(ImageSG.prompt)
    await query.answer()
    await message.answer("Опишите изображение.\n/cancel — отмена")


async def sg_image_prompt(
    message: Message,
    state: FSMContext,
    assistant: AssistAIClient,
    assistant_storage: AssistantStorage,
    topic_storage: TopicHistoryStorage,
) -> None:
    """Генерирует картинку по описанию и выбору с экрана."""
    prompt = (message.text or "").strip()
    if not prompt:
        await message.answer("Опишите изображение.\n/cancel — отмена")
        return
    if not security.validate_prompt_length(prompt):
        await message.answer("⚠️ Описание слишком длинное. Максимум 3000 символов.")
        return
    user_id = message.from_user.id if message.from_user else message.chat.id
    if not await rate_limiter.check_image_limit(user_id):
        retry_after = await rate_limiter.get_retry_after(user_id, "image")
        await message.answer(
            f"⏳ Лимит генерации изображений. Подождите {int(retry_after) or 60} секунд."
        )
        logger.warning("rate_limit_exceeded", user_id=user_id, operation="image")
        return
    data = await state.get_data()
    model = M._optional_model(data.get("image_model"))
    model_id = model or assistant.config.default_image_model
    size = data.get("image_size")
    if not isinstance(size, str) or size not in M._size_choices(model_id):
        size = M._default_image_size(model_id)
    await R._reset_flow(state)
    menu = await R._menu_for(message.chat.id, assistant_storage, topic_storage)
    await R._generate_and_send(
        message,
        assistant,
        security.sanitize_prompt(prompt),
        model,
        size,
        reply_markup=menu,
    )
