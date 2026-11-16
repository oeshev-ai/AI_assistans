"""Снятые текстовые id сохраняют выбор: и в профиле, и у vision."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.bot.models import _vision_model_of
from app.llm.models_config import LEGACY_TEXT_MODEL_ALIASES
from app.storage.assistant_storage import DEFAULT_PERSONA_PROMPT, AssistantProfile
from bot import _text_model_of


def _assistant() -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(default_text_model="deepseek/deepseek-v4.1-flash")
    )


def _profile(
    *,
    text_model: str | None = None,
    vision_model: str | None = None,
) -> AssistantProfile:
    return AssistantProfile(
        name="Старый",
        system_prompt=DEFAULT_PERSONA_PROMPT,
        text_model=text_model,
        vision_model=vision_model,
    )


@pytest.mark.parametrize("old", sorted(LEGACY_TEXT_MODEL_ALIASES))
def test_text_ids_alias_on_read(old: str) -> None:
    profile = _profile(text_model=old)
    assert profile.text_model == LEGACY_TEXT_MODEL_ALIASES[old]
    assert (
        _text_model_of(profile, _assistant(), chat_id=1)  # type: ignore[arg-type]
        == LEGACY_TEXT_MODEL_ALIASES[old]
    )


def test_vision_model_aliases_too() -> None:
    profile = _profile(vision_model="anthropic/claude-haiku-5-5")
    assert profile.vision_model == "google/gemini-3.8-flash"
    assert (
        _vision_model_of(profile, _assistant(), chat_id=2)  # type: ignore[arg-type]
        == "google/gemini-3.8-flash"
    )


def test_unknown_text_id_falls_back_to_default() -> None:
    profile = _profile(text_model="vendor/removed-text-model")
    assert profile.text_model == "vendor/removed-text-model"
    assistant = _assistant()
    assert (
        _text_model_of(profile, assistant, chat_id=3)  # type: ignore[arg-type]
        == assistant.config.default_text_model
    )


def test_text_alias_flag_off_keeps_id_until_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEXT_LEGACY_ALIASES", "0")
    profile = _profile(text_model="qwen/qwen3.8-flash")
    assert profile.text_model == "qwen/qwen3.8-flash"
    assistant = _assistant()
    assert (
        _text_model_of(profile, assistant, chat_id=4)  # type: ignore[arg-type]
        == assistant.config.default_text_model
    )
