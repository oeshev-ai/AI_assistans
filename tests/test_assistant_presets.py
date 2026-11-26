"""Каталог персон и иконки."""

from __future__ import annotations

from app.llm.models_config import (
    AVAILABLE_IMAGE_MODELS,
    AVAILABLE_TEXT_MODELS,
    AVAILABLE_TTS_MODELS,
    SPEECHKIT_TTS_MODEL,
)
from app.llm.speechkit_tts import SPEECHKIT_VOICES
from app.storage.assistant_presets import (
    ASSISTANT_PRESETS,
    AssistantPreset,
    _size_ok,
    _tts_voices_for_model,
    get_preset,
    preset_profile,
    validate_catalog,
)
from app.storage.assistant_storage import DEFAULT_PERSONA_PROMPT, AssistantProfile
from app.storage.persona_icons import DEFAULT_ICON, normalize_icon, persona_label


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


def test_openai_size_aliases_stay_valid() -> None:
    assert _size_ok("openai/gpt-image-2.6-flare", "1024x1792")
    assert _size_ok("openai/gpt-image-2.6-flare", "1792x1024")
    assert _size_ok("openai/gpt-image-2.6-flare", "1024x1536")


def test_preset_card_marks_vision_on_the_text_model() -> None:
    from types import SimpleNamespace

    from app.bot.screens.presets import _preset_preview_parts

    assistant = SimpleNamespace(
        config=SimpleNamespace(
            default_text_model="openai/gpt-6-luna",
            default_image_model="openai/gpt-image-2.6-flare",
            default_tts_model="openai/gpt-4o-mini-tts",
        )
    )
    helper = get_preset("helper")
    eco = get_preset("eco")
    assert helper is not None and eco is not None
    helper_text, _markup = _preset_preview_parts(helper, assistant)  # type: ignore[arg-type]
    eco_text, _markup = _preset_preview_parts(eco, assistant)  # type: ignore[arg-type]
    assert "Vision: +" in helper_text
    assert "Vision: +" in eco_text


def test_helper_generates_on_flux_klein() -> None:
    preset = get_preset("helper")
    assert preset is not None
    assert preset.image_model == "black-forest-labs/flux-3-image-klein"
    assert preset.image_size == "1K"


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


def test_speechkit_preset_uses_its_voices_and_opus() -> None:
    voices = _tts_voices_for_model(SPEECHKIT_TTS_MODEL)
    assert voices == SPEECHKIT_VOICES
    assert "marina" in voices
    assert "nova" not in voices
    assert _tts_voices_for_model("openai/gpt-4o-mini-tts") == (
        "alloy",
        "verse",
        "nova",
        "shimmer",
    )
    preset = AssistantPreset(
        id="speaker",
        icon="🎵",
        name="Диктор",
        summary="Озвучка SpeechKit",
        system_prompt="Ты диктор. Говори ясно и коротко, без лишних вступлений.",
        tts_model=SPEECHKIT_TTS_MODEL,
        tts_voice="marina",
    )
    assert preset.tts_voice in _tts_voices_for_model(preset.tts_model or "")
    profile = preset_profile(preset, [])
    assert profile.tts_model == SPEECHKIT_TTS_MODEL
    assert profile.tts_voice == "marina"
    assert profile.tts_format == "opus"
    tutor = get_preset("tutor")
    assert tutor is not None
    assert preset_profile(tutor, []).tts_format is None


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
