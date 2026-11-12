"""Каталог моделей и MODEL_HINTS."""

from __future__ import annotations

from app.llm.models_config import (
    AVAILABLE_IMAGE_MODELS,
    AVAILABLE_TEXT_MODELS,
    AVAILABLE_TTS_MODELS,
    MODEL_HINTS,
    model_hint,
    validate_hints,
)


def test_validate_hints_passes() -> None:
    validate_hints()


def test_every_catalog_model_has_hint_within_limit() -> None:
    for model_id in (*AVAILABLE_TEXT_MODELS, *AVAILABLE_IMAGE_MODELS):
        hint = MODEL_HINTS[model_id]
        assert 1 <= len(hint) <= 60


def test_model_hints_exclude_tts_and_extras() -> None:
    for model_id in AVAILABLE_TTS_MODELS:
        assert model_id not in MODEL_HINTS
    catalog = set(AVAILABLE_TEXT_MODELS) | set(AVAILABLE_IMAGE_MODELS)
    assert set(MODEL_HINTS) == catalog


def test_model_hint_empty_for_unknown() -> None:
    assert model_hint(None) == ""
    assert model_hint("unknown/model") == ""


def test_text_catalog_uses_new_ids() -> None:
    assert "openai/gpt-6-sol" in AVAILABLE_TEXT_MODELS
    assert "openai/gpt-6.1-terra" in AVAILABLE_TEXT_MODELS
    assert "openai/gpt-6.1-sol" not in AVAILABLE_TEXT_MODELS
    assert "openai/gpt-6-astra" not in AVAILABLE_TEXT_MODELS
