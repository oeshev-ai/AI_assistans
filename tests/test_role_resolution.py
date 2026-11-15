"""Приоритет ролей: кнопка важнее текста, без обоих — модель решает."""

from __future__ import annotations

from app.bot.photo_routing import resolve_roles


def test_override_beats_caption() -> None:
    roles = resolve_roles(
        "сделай фото 1 в стиле фото 2",
        2,
        base_override=1,
    )
    assert roles == (1, [0])


def test_text_when_no_override() -> None:
    assert resolve_roles("предмет с фото 1 помести на фото 2", 2) == (1, [0])


def test_unknown_without_override_or_text() -> None:
    assert resolve_roles("сделай красиво", 3) is None


def test_override_out_of_range() -> None:
    assert resolve_roles("фото 1", 2, base_override=5) is None
