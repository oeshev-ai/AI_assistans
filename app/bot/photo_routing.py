"""Чистое решение: намерение по фото и явные роли основы/референсов.

Индексы нулевые. Номера фото в тексте — с единицы.
Порядок прихода альбома роли не задаёт.
"""

from __future__ import annotations

import re
from typing import Literal

PhotoIntent = Literal["edit", "analyze", "ask"]

_EDIT_MARKERS = (
    "замени",
    "поменяй",
    "надень",
    "помести",
    "убери",
    "добавь",
    "вставь",
    "перерисуй",
    "отредактир",
    "в стиле",
    "костюм",
    "сделай фото",
    "перенеси",
)
_ANALYZE_MARKERS = (
    "что это",
    "опиши",
    "расскажи",
    "проанализ",
    "распознай",
    "что на фото",
    "что изображ",
    "что здесь",
)
_PHOTO_REF = re.compile(r"фото\s*(\d+)|(?<![0-9a-zа-я])с\s+(\d+)", re.IGNORECASE)
_STYLE = re.compile(
    r"фото\s*(\d+).{0,80}?в стиле(?:\s+фото)?\s*(\d+)",
    re.IGNORECASE | re.DOTALL,
)
_PERSON = re.compile(r"человек\s+с\s+(?:фото\s*)?(\d+)", re.IGNORECASE)


def _photo_intent(text: str) -> PhotoIntent:
    """edit важнее analyze. Пустая подпись — ask, без вызова модели."""
    cleaned = text.strip().casefold()
    if not cleaned:
        return "ask"
    if any(marker in cleaned for marker in _EDIT_MARKERS):
        return "edit"
    if any(marker in cleaned for marker in _ANALYZE_MARKERS):
        return "analyze"
    return "ask"


def _mentioned_indexes(text: str, count: int) -> list[int]:
    """Номера фото в порядке упоминания, без повторов, в диапазоне альбома."""
    found: list[int] = []
    for match in _PHOTO_REF.finditer(text):
        raw = match.group(1) or match.group(2)
        number = int(raw)
        index = number - 1
        if index < 0 or index >= count or index in found:
            continue
        found.append(index)
    return found


def _explicit_roles(text: str, count: int) -> tuple[int, list[int]] | None:
    """Роли из формулировки. None — в тексте нет явной основы.

    A–D: последнее упомянутое фото — основа, предыдущие — референсы.
    E: «фото N в стиле фото M» — основа N, референс M.
    F: «человек с N» — основа, остальные упомянутые — референсы.
    """
    if count < 1 or not text.strip():
        return None
    style = _STYLE.search(text)
    if style:
        base = int(style.group(1)) - 1
        ref = int(style.group(2)) - 1
        if 0 <= base < count and 0 <= ref < count and base != ref:
            return base, [ref]
    person = _PERSON.search(text)
    if person:
        base = int(person.group(1)) - 1
        if 0 <= base < count:
            mentioned = _mentioned_indexes(text, count)
            refs = [index for index in mentioned if index != base]
            if not refs and count > 1:
                refs = [index for index in range(count) if index != base]
            return base, refs
    mentioned = _mentioned_indexes(text, count)
    if len(mentioned) < 2:
        return None
    base = mentioned[-1]
    refs = [index for index in mentioned[:-1] if index != base]
    if not refs:
        return None
    return base, refs


def resolve_roles(
    text: str,
    count: int,
    *,
    base_override: int | None = None,
) -> tuple[int, list[int]] | None:
    """Приоритет: явный выбор кнопкой, затем разбор текста."""
    if count < 1:
        return None
    if base_override is not None:
        if not 0 <= base_override < count:
            return None
        refs = [index for index in range(count) if index != base_override]
        return base_override, refs
    return _explicit_roles(text, count)
