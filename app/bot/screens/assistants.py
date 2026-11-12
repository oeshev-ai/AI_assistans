"""Screen builders."""

from __future__ import annotations

from app.bot import models as M
from app.bot.constants import (
    PAGE_SIZE,
    AssistantProfile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    persona_label,
)


def _assistants_text(
    profiles: list[AssistantProfile], active_id: str, page: int = 0
) -> str:
    """Текст экрана персон: кто активен и номер страницы. Имена только на кнопках."""
    current, total = M._page_bounds(len(profiles), page)
    active = next((item for item in profiles if item.id == active_id), None)
    active_name = persona_label(active.name, active.icon) if active is not None else "—"
    lines = [f"Активный: ✅ {active_name}"]
    if not profiles:
        lines.append("")
        lines.append("Персон пока нет.")
    lines.append("")
    lines.append(f"Стр. {current + 1}/{total}")
    return "\n".join(lines)


def _assistants_keyboard(
    profiles: list[AssistantProfile], active_id: str, page: int = 0
) -> InlineKeyboardMarkup:
    """Кнопки персон текущей страницы и переход по страницам."""
    current, total = M._page_bounds(len(profiles), page)
    chunk = profiles[current * PAGE_SIZE : (current + 1) * PAGE_SIZE]
    rows: list[list[InlineKeyboardButton]] = []
    for profile in chunk:
        mark = "✅ " if profile.id == active_id else ""
        rows.append(
            [
                InlineKeyboardButton(
                    text=M._clip_label(mark, persona_label(profile.name, profile.icon)),
                    callback_data=f"asst:use:{profile.id}",
                )
            ]
        )
    pager = M._pager_row(current, total, "asst_page")
    if pager is not None:
        rows.append(pager)
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _assistants_markup(
    profiles: list[AssistantProfile], active_id: str, page: int = 0
) -> tuple[str, InlineKeyboardMarkup]:
    """Текст и кнопки экрана персон. Здесь только переключение."""
    return (
        _assistants_text(profiles, active_id, page),
        _assistants_keyboard(profiles, active_id, page),
    )
