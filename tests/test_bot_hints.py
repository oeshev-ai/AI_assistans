"""Подсказки моделей и soft-fallback устаревших id."""

from __future__ import annotations

from types import SimpleNamespace

from app.bot.constants import TELEGRAM_TEXT_LIMIT
from app.bot.models import (
    _fit_screen,
    _hints_block,
    _image_model_of,
    _model_keyboard,
    _text_model_of,
)
from app.bot.screens.draft import _icons_keyboard
from app.llm.models_config import (
    AVAILABLE_IMAGE_MODELS,
    AVAILABLE_TEXT_MODELS,
    model_hint,
)
from app.storage.assistant_storage import DEFAULT_PERSONA_PROMPT, AssistantProfile


def _assistant() -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            default_text_model="qwen/qwen3.8-flash",
            default_image_model="openai/gpt-image-2.6-flare",
        )
    )


def test_hints_block_marks_current_and_includes_hints() -> None:
    current = AVAILABLE_TEXT_MODELS[0]
    block = _hints_block(AVAILABLE_TEXT_MODELS, current)
    assert f"✅ {current} — {model_hint(current)}" in block
    for model_id in AVAILABLE_TEXT_MODELS[1:]:
        assert f"• {model_id} — {model_hint(model_id)}" in block


def test_text_model_keyboard_closes_with_done() -> None:
    keyboard = _model_keyboard(AVAILABLE_TEXT_MODELS, "set_text", done="picker:done")
    callbacks = [
        button.callback_data for row in keyboard.inline_keyboard for button in row
    ]
    assert callbacks[-1] == "picker:done"
    assert all(
        item is not None and item.startswith("set_text:") for item in callbacks[:-1]
    )


def test_draft_icon_done_stays_on_wizard_callback() -> None:
    keyboard = _icons_keyboard("asst:new_icon", "🤖", done="asst:new_icon_back")
    callbacks = [
        button.callback_data for row in keyboard.inline_keyboard for button in row
    ]
    assert "picker:done" not in callbacks
    assert callbacks[-1] == "asst:new_icon_back"


async def test_settings_navigation_removes_open_picker() -> None:
    from app.bot.constants import _SCREEN_PICKER
    from app.bot.runtime import _dismiss_open_picker

    deleted: list[tuple[int, int]] = []

    class _Bot:
        async def delete_message(self, *, chat_id: int, message_id: int) -> None:
            deleted.append((chat_id, message_id))

    class _Chat:
        id = 9

    class _Message:
        bot = _Bot()
        message_id = 1
        chat = _Chat()

    class _State:
        def __init__(self) -> None:
            self.data = {_SCREEN_PICKER: 2}

        async def get_data(self) -> dict[str, int]:
            return self.data

        async def set_data(self, data: dict[str, int]) -> None:
            self.data = data

    state = _State()
    await _dismiss_open_picker(_Message(), state)  # type: ignore[arg-type]
    assert deleted == [(9, 2)]
    assert _SCREEN_PICKER not in state.data


def test_persona_icon_done_closes_picker() -> None:
    keyboard = _icons_keyboard("set_icon", "🤖")
    callbacks = [
        button.callback_data for row in keyboard.inline_keyboard for button in row
    ]
    assert callbacks[-1] == "picker:done"


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
