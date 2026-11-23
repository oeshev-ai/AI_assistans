"""intent=generate with reference photos keeps edit API but generate-tier params."""

from __future__ import annotations

from app.bot.photo_flow import _generate_with_refs_model


def test_openai_generate_model_preferred_for_refs() -> None:
    generate = "openai/gpt-image-2.6-flare"
    edit = "openai/gpt-image-2.6-flare"
    assert _generate_with_refs_model(generate, edit) == generate


def test_flux_generate_falls_back_to_edit_model() -> None:
    generate = "black-forest-labs/flux-3-image-klein"
    edit = "openai/gpt-image-2.6-flare"
    assert _generate_with_refs_model(generate, edit) == edit
