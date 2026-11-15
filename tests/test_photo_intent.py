"""Намерение по подписи фото: правка важнее описания, пусто — вопрос."""

from __future__ import annotations

from app.bot.photo_routing import _photo_intent


def test_empty_caption_asks() -> None:
    assert _photo_intent("") == "ask"
    assert _photo_intent("   ") == "ask"


def test_question_is_analyze() -> None:
    assert _photo_intent("что это?") == "analyze"
    assert _photo_intent("Опиши, пожалуйста") == "analyze"


def test_edit_beats_analyze() -> None:
    assert _photo_intent("опиши и замени фон") == "edit"
    assert _photo_intent("надень костюм с фото 1") == "edit"


def test_unrelated_text_asks() -> None:
    assert _photo_intent("привет") == "ask"
