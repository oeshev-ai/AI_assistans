"""Каталог персон и иконки."""

from __future__ import annotations

from assistant_presets import (
    ASSISTANT_PRESETS,
    get_preset,
    preset_profile,
    validate_catalog,
)
from assistant_storage import DEFAULT_PERSONA_PROMPT, AssistantProfile
from models_config import (
    AVAILABLE_IMAGE_MODELS,
    AVAILABLE_TEXT_MODELS,
    AVAILABLE_TTS_MODELS,
)
from persona_icons import DEFAULT_ICON, normalize_icon, persona_label


def test_validate_catalog_passes() -> None:
    validate_catalog()
    assert len(ASSISTANT_PRESETS) == 10


def test_preset_models_in_catalogs() -> None:
    for preset in ASSISTANT_PRESETS:
        if preset.text_model is not None:
            assert preset.text_model in AVAILABLE_TEXT_MODELS
        if preset.image_model is not None:
            assert preset.image_model in AVAILABLE_IMAGE_MODELS
        if preset.tts_model is not None:
            assert preset.tts_model in AVAILABLE_TTS_MODELS


def test_get_preset() -> None:
    assert get_preset("dev") is not None
    assert get_preset("missing") is None


def test_preset_profile_new_id_and_dedup() -> None:
    preset = get_preset("helper")
    assert preset is not None
    first = preset_profile(preset, [])
    assert first.id != preset.id
    assert first.icon == preset.icon
    assert first.name == preset.name
    second = preset_profile(preset, [first.name])
    assert second.name == f"{preset.name} (2)"
    assert len(second.name) <= 50
    third = preset_profile(preset, [first.name, second.name])
    assert third.name == f"{preset.name} (3)"


def test_preset_profile_long_name_truncates_with_suffix() -> None:
    preset = get_preset("dev")
    assert preset is not None
    long_taken = "Сеньор-разработчик"
    profile = preset_profile(preset, [long_taken])
    assert profile.name.endswith(" (2)")
    assert len(profile.name) <= 50


def test_old_profile_without_icon_defaults() -> None:
    profile = AssistantProfile.model_validate(
        {
            "id": "abc12345",
            "name": "Старый",
            "system_prompt": DEFAULT_PERSONA_PROMPT,
        }
    )
    assert profile.icon == "🤖"


def test_empty_icon_normalized() -> None:
    profile = AssistantProfile.model_validate(
        {
            "id": "abc12345",
            "name": "Тест",
            "system_prompt": DEFAULT_PERSONA_PROMPT,
            "icon": "",
        }
    )
    assert profile.icon == "🤖"


def test_persona_label_and_normalize_icon() -> None:
    assert normalize_icon(None) == DEFAULT_ICON
    assert normalize_icon("???") == DEFAULT_ICON
    assert normalize_icon("💻") == "💻"
    assert persona_label("Dev", "💻") == "💻 Dev"
    assert persona_label("Dev", None) == f"{DEFAULT_ICON} Dev"
