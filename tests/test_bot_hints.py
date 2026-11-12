"""Подсказки моделей и soft-fallback устаревших id в bot.py."""

from __future__ import annotations

from types import SimpleNamespace

from app.llm.models_config import (
    AVAILABLE_IMAGE_MODELS,
    AVAILABLE_TEXT_MODELS,
    model_hint,
)
from app.storage.assistant_storage import DEFAULT_PERSONA_PROMPT, AssistantProfile
from bot import (
    TELEGRAM_TEXT_LIMIT,
    _fit_screen,
    _hints_block,
    _image_model_of,
    _text_model_of,
)


def _assistant() -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            default_text_model="qwen/qwen3.8-flash",
            default_image_model="openai/gpt-image-1-mini",
        )
    )


def test_hints_block_marks_current_and_includes_hints() -> None:
    current = AVAILABLE_TEXT_MODELS[0]
    block = _hints_block(AVAILABLE_TEXT_MODELS, current)
    assert f"✅ {current} — {model_hint(current)}" in block
    for model_id in AVAILABLE_TEXT_MODELS[1:]:
        assert f"• {model_id} — {model_hint(model_id)}" in block


def test_fit_screen_truncates_to_telegram_limit() -> None:
    long_text = "a" * (TELEGRAM_TEXT_LIMIT + 50)
    fitted = _fit_screen(long_text)
    assert len(fitted) == TELEGRAM_TEXT_LIMIT
    assert fitted.endswith("…")
    assert _fit_screen("short") == "short"


def test_text_model_of_stale_falls_back_to_default() -> None:
    assistant = _assistant()
    stale = AssistantProfile(
        name="Test",
        system_prompt=DEFAULT_PERSONA_PROMPT,
        text_model="vendor/removed-text-model",
    )
    assert (
        _text_model_of(stale, assistant, chat_id=1)  # type: ignore[arg-type]
        == assistant.config.default_text_model
    )
    valid = AssistantProfile(
        name="Test",
        system_prompt=DEFAULT_PERSONA_PROMPT,
        text_model="openai/gpt-6-sol",
    )
    assert _text_model_of(valid, assistant) == "openai/gpt-6-sol"  # type: ignore[arg-type]


def test_image_model_of_stale_falls_back_to_default() -> None:
    assistant = _assistant()
    stale = AssistantProfile(
        name="Test",
        system_prompt=DEFAULT_PERSONA_PROMPT,
        image_model="vendor/removed-image-model",
    )
    assert (
        _image_model_of(stale, assistant, chat_id=2)  # type: ignore[arg-type]
        == assistant.config.default_image_model
    )
    valid_id = AVAILABLE_IMAGE_MODELS[0]
    valid = AssistantProfile(
        name="Test",
        system_prompt=DEFAULT_PERSONA_PROMPT,
        image_model=valid_id,
    )
    assert _image_model_of(valid, assistant) == valid_id  # type: ignore[arg-type]
