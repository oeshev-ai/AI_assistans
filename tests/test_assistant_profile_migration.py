"""Старые профили без голосовых полей читаются с дефолтами."""

from __future__ import annotations

from assistant_storage import (
    DEFAULT_PERSONA_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    AssistantProfile,
    compose_system_prompt,
    default_assistant_profile,
)


def test_old_profile_gets_voice_defaults() -> None:
    profile = AssistantProfile.model_validate(
        {
            "id": "abc12345",
            "name": "Старый",
            "system_prompt": DEFAULT_PERSONA_PROMPT,
            "text_model": "openai/gpt-4o-mini",
            "image_model": None,
            "image_size": None,
        }
    )

    assert profile.voice_mode == "auto"
    assert profile.tts_model is None
    assert profile.tts_voice is None
    assert profile.tts_format is None
    assert profile.stt_language is None
    assert profile.vision_model is None


def test_compose_system_prompt_appends_shared_rules() -> None:
    composed = compose_system_prompt("Ты строгий репетитор математики.")
    assert composed.startswith("Ты строгий репетитор математики.")
    assert DEFAULT_SYSTEM_PROMPT in composed
    assert "generate_image" in composed


def test_compose_system_prompt_skips_duplicate_shared() -> None:
    already = f"Кастом.\n\n{DEFAULT_SYSTEM_PROMPT}"
    assert compose_system_prompt(already) == already


def test_default_profile_stores_persona_only() -> None:
    profile = default_assistant_profile()
    assert profile.system_prompt == DEFAULT_PERSONA_PROMPT
    assert DEFAULT_SYSTEM_PROMPT not in profile.system_prompt
    assert DEFAULT_SYSTEM_PROMPT in compose_system_prompt(profile.system_prompt)
