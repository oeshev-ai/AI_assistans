"""Screen builders."""

from __future__ import annotations

from app.bot import models as M
from app.bot.constants import *


def _draft_models_text(data: dict[str, Any], assistant: AssistAIClient) -> str:
    """Текущий выбор моделей на шаге создания персоны."""
    text_model = M._optional_model(data.get("flow_text_model"))
    image_model = M._optional_model(data.get("flow_image_model"))
    icon = normalize_icon(
        data.get("flow_icon") if isinstance(data.get("flow_icon"), str) else None
    )
    text_current = text_model or assistant.config.default_text_model
    image_current = image_model or assistant.config.default_image_model
    text_line = (
        f"{assistant.config.default_text_model} (по умолчанию)"
        if text_model is None
        else text_model
    )
    image_line = (
        f"{assistant.config.default_image_model} (по умолчанию)"
        if image_model is None
        else image_model
    )
    return M._fit_screen(
        "Выберите модели (необязательно) или нажмите «Готово».\n"
        f"Иконка: {icon}\n"
        f"Текст: {text_line}\n"
        f"{M._hints_block(AVAILABLE_TEXT_MODELS, text_current)}\n"
        f"Картинки: {image_line}\n"
        f"{M._hints_block(AVAILABLE_IMAGE_MODELS, image_current)}"
    )


def _draft_models_keyboard(data: dict[str, Any]) -> InlineKeyboardMarkup:
    """Кнопки временного выбора моделей и подтверждение создания."""
    text_model = M._optional_model(data.get("flow_text_model"))
    image_model = M._optional_model(data.get("flow_image_model"))
    icon = normalize_icon(
        data.get("flow_icon") if isinstance(data.get("flow_icon"), str) else None
    )
    rows: list[list[InlineKeyboardButton]] = [
        [
            InlineKeyboardButton(
                text=f"Иконка: {icon}",
                callback_data="asst:new_icon_menu",
            )
        ]
    ]
    for index, model in enumerate(AVAILABLE_TEXT_MODELS):
        mark = "✅ " if text_model == model else ""
        rows.append(
            [
                InlineKeyboardButton(
                    text=M._clip_label(mark, model),
                    callback_data=f"asst:new_text:{index}",
                )
            ]
        )
    for index, model in enumerate(AVAILABLE_IMAGE_MODELS):
        mark = "✅ " if image_model == model else ""
        rows.append(
            [
                InlineKeyboardButton(
                    text=M._clip_label(mark, model, suffix=" 🖼"),
                    callback_data=f"asst:new_image:{index}",
                )
            ]
        )
    rows.append(
        [
            InlineKeyboardButton(
                text="Готово",
                callback_data="asst:new_commit",
                style="success",
            )
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _icons_keyboard(prefix: str, current: str) -> InlineKeyboardMarkup:
    """Сетка иконок по 5 в ряд; callback — индекс в PERSONA_ICONS."""
    current_icon = normalize_icon(current)
    rows: list[list[InlineKeyboardButton]] = []
    row: list[InlineKeyboardButton] = []
    for index, icon in enumerate(PERSONA_ICONS):
        mark = "✅ " if icon == current_icon else ""
        row.append(
            InlineKeyboardButton(
                text=f"{mark}{icon}",
                callback_data=f"{prefix}:{index}",
            )
        )
        if len(row) >= 5:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append(
        [
            InlineKeyboardButton(
                text="Готово", callback_data="picker:done", style="success"
            )
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)
