"""Иконки персон для кнопок и заголовков."""

from __future__ import annotations

DEFAULT_ICON = "🤖"
PERSONA_ICONS: tuple[str, ...] = (
    "🤖",
    "💻",
    "✍️",
    "📊",
    "🎓",
    "🌍",
    "🎨",
    "⚡",
    "🔍",
    "🧙",
    "🧠",
    "📚",
    "🩺",
    "⚖️",
    "🍳",
    "🏋️",
    "✈️",
    "🎬",
    "🎵",
    "🛠️",
)


def normalize_icon(raw: str | None) -> str:
    """Пустое/неизвестное значение → DEFAULT_ICON. Берётся из PERSONA_ICONS."""
    cleaned = (raw or "").strip()
    if cleaned in PERSONA_ICONS:
        return cleaned
    return DEFAULT_ICON


def persona_label(name: str, icon: str | None) -> str:
    """Строка для кнопок и заголовков: ``💻 Сеньор-разработчик``."""
    return f"{normalize_icon(icon)} {name.strip()}".strip()
