"""Явные роли A–F. Индексы с нуля, номера фото в тексте — с единицы."""

from __future__ import annotations

import pytest

from app.bot.photo_routing import _explicit_roles


@pytest.mark.parametrize(
    ("text", "count", "expected"),
    [
        ("замени фон на фото 1 изображением с фото 2", 2, (1, [0])),
        ("костюм с фото 1 надень на человека с фото 2", 2, (1, [0])),
        ("лицо с фото 1 замени лицом с фото 2", 2, (1, [0])),
        ("предмет с фото 1 помести на фото 2", 2, (1, [0])),
        ("сделай фото 1 в стиле фото 2", 2, (0, [1])),
        ("человек с 1, костюм с 2, фон с 3", 3, (0, [1, 2])),
    ],
)
def test_scenarios(text: str, count: int, expected: tuple[int, list[int]]) -> None:
    assert _explicit_roles(text, count) == expected


def test_no_roles_without_two_photos() -> None:
    assert _explicit_roles("просто привет", 2) is None
    assert _explicit_roles("фото 1", 2) is None
