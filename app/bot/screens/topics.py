"""Screen builders."""

from __future__ import annotations

from app.bot import models as M
from app.bot.constants import (
    PAGE_SIZE,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)


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
    topics: list[dict[str, str]],
    active: str,
    page: int = 0,
    *,
    assistant_id: str,
) -> InlineKeyboardMarkup:
    """Кнопки тем текущей страницы и «Готово».

    Id персоны в callback действий, не «кто сейчас активен».
    """
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
                    callback_data=f"topic:use:{assistant_id}:{topic_id}",
                ),
                InlineKeyboardButton(
                    text="✏️",
                    callback_data=f"topic:rename:{assistant_id}:{topic_id}",
                ),
                InlineKeyboardButton(
                    text="🗑",
                    callback_data=f"topic_del_ask:{assistant_id}:{topic_id}",
                ),
            ]
        )
    rows.append(
        [
            InlineKeyboardButton(
                text="Новая тема",
                callback_data=f"topic:new:{assistant_id}",
            )
        ]
    )
    pager = M._pager_row(current, total, f"topic_page:{assistant_id}")
    if pager is not None:
        rows.append(pager)
    rows.append(
        [
            InlineKeyboardButton(
                text="Готово", callback_data="topic:done", style="success"
            )
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _topics_markup(
    topics: list[dict[str, str]],
    active_id: str,
    page: int = 0,
    *,
    assistant_id: str,
) -> tuple[str, InlineKeyboardMarkup]:
    """Текст и кнопки экрана тем."""
    return (
        _topics_text(topics, active_id, page),
        _topics_keyboard(topics, active_id, page, assistant_id=assistant_id),
    )
