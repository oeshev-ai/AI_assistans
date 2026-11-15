"""Экран действия по фото и выбор основы."""

from __future__ import annotations

from app.bot.constants import InlineKeyboardButton, InlineKeyboardMarkup


def _photo_action_view(
    count: int, base_override: int | None
) -> tuple[str, InlineKeyboardMarkup]:
    """Кнопки: правка, основа (если фото больше одного), описание, вопрос, отмена."""
    base_label = "авто" if base_override is None else f"фото {base_override + 1}"
    text = f"Фото: {count}. Основа: {base_label}.\nЧто сделать?"
    rows: list[list[InlineKeyboardButton]] = [
        [InlineKeyboardButton(text="Изменить", callback_data="photo:edit")],
    ]
    if count > 1:
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"Основа: {base_label}",
                    callback_data="photo:base_next",
                )
            ]
        )
    rows.append([InlineKeyboardButton(text="Описать", callback_data="photo:analyze")])
    rows.append([InlineKeyboardButton(text="Задать вопрос", callback_data="photo:ask")])
    rows.append([InlineKeyboardButton(text="Отмена", callback_data="photo:cancel")])
    return text, InlineKeyboardMarkup(inline_keyboard=rows)


def _photo_base_pick_view(count: int) -> tuple[str, InlineKeyboardMarkup]:
    """Один повтор: пользователь сам указывает основу."""
    rows = [
        [
            InlineKeyboardButton(
                text=f"Фото {index + 1}",
                callback_data=f"photo:base_set:{index}",
            )
        ]
        for index in range(count)
    ]
    rows.append([InlineKeyboardButton(text="Отмена", callback_data="photo:cancel")])
    return "Какое фото основа?", InlineKeyboardMarkup(inline_keyboard=rows)
