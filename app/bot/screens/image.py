"""Screen builders."""

from __future__ import annotations

from app.bot import models as M
from app.bot.constants import *


def _image_view(
    selected: str | None, size: str, assistant: AssistAIClient
) -> tuple[str, InlineKeyboardMarkup]:
    """Экран генерации: модель, размер семьи модели и переход к описанию."""
    label = selected or assistant.config.default_image_model
    allowed = M._size_choices(label)
    shown = size if size in allowed else M._default_image_size(label)
    text = M._fit_screen(
        f"Генерация изображения\nМодель: {label}\nРазмер: {shown}\n\n"
        f"{M._hints_block(AVAILABLE_IMAGE_MODELS, label)}"
    )
    rows: list[list[InlineKeyboardButton]] = []
    for index, model in enumerate(AVAILABLE_IMAGE_MODELS):
        mark = "✅ " if selected == model else ""
        rows.append(
            [
                InlineKeyboardButton(
                    text=M._clip_label(mark, model),
                    callback_data=f"image:model:{index}",
                )
            ]
        )
    size_row: list[InlineKeyboardButton] = []
    for item in allowed:
        mark = "✅ " if item == shown else ""
        size_row.append(
            InlineKeyboardButton(
                text=f"{mark}{item}", callback_data=f"image:size:{item}"
            )
        )
    rows.append(size_row)
    rows.append(
        [InlineKeyboardButton(text="✍️ Описать картинку", callback_data="image:go")]
    )
    return text, InlineKeyboardMarkup(inline_keyboard=rows)
