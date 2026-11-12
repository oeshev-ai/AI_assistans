"""Screen builders."""

from __future__ import annotations

from app.bot import models as M
from app.bot.constants import *


def _topics_text(topics: list[dict[str, str]], active: str, page: int = 0) -> str:
    """Текст экрана тем: какая активна и номер страницы. Имена только на кнопках."""
    current, total = M._page_bounds(len(topics), page)
    active_name = next(
        (
            M._topic_display_name(item["name"])
            for item in topics
            if item["id"] == active
        ),
        "нет",
    )
    lines = [f"Активная: 📍 {active_name}"]
    if not topics:
        lines.append("")
        lines.append("Тем пока нет.")
    lines.append("")
    lines.append(f"Стр. {current + 1}/{total}")
    return "\n".join(lines)


def _topics_keyboard(
    topics: list[dict[str, str]], active: str, page: int = 0
) -> InlineKeyboardMarkup:
    """Кнопки тем текущей страницы: переключить, переименовать, удалить."""
    current, total = M._page_bounds(len(topics), page)
    chunk = topics[current * PAGE_SIZE : (current + 1) * PAGE_SIZE]
    rows: list[list[InlineKeyboardButton]] = []
    for topic in chunk:
        name = M._topic_display_name(topic["name"])
        topic_id = topic["id"]
        mark = "✅ " if topic_id == active else ""
        rows.append(
            [
                InlineKeyboardButton(
                    text=M._clip_label(mark, name),
                    callback_data=f"topic:use:{topic_id}",
                ),
                InlineKeyboardButton(
                    text="✏️", callback_data=f"topic:rename:{topic_id}"
                ),
                InlineKeyboardButton(
                    text="🗑", callback_data=f"topic_del_ask:{topic_id}"
                ),
            ]
        )
    rows.append([InlineKeyboardButton(text="Новая тема", callback_data="topic:new")])
    pager = M._pager_row(current, total, "topic_page")
    if pager is not None:
        rows.append(pager)
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _topics_markup(
    topics: list[dict[str, str]], active_id: str, page: int = 0
) -> tuple[str, InlineKeyboardMarkup]:
    """Текст и кнопки экрана тем."""
    return (
        _topics_text(topics, active_id, page),
        _topics_keyboard(topics, active_id, page),
    )
