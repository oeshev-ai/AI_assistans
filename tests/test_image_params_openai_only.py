"""quality уходит только моделям OpenAI. complexity не отправляется."""

from __future__ import annotations

from app.llm.image_ops import _image_generation_body


def test_legacy_pixel_sizes_map_to_current() -> None:
    body = _image_generation_body("openai/gpt-image-2.6-flare", "poster", "1024x1792")
    assert body["size"] == "1024x1536"
    wide = _image_generation_body("openai/gpt-image-2.6-flare", "poster", "1792x1024")
    assert wide["size"] == "1536x1024"


def test_openai_gets_quality_without_complexity() -> None:
    body = _image_generation_body(
        "openai/gpt-image-2.6-flare",
        "poster",
        "1024x1024",
        quality="high",
    )
    assert body["quality"] == "high"
    assert "complexity" not in body


def test_other_families_omit_openai_params() -> None:
    for model, size in (
        ("black-forest-labs/flux-3-image", "1K"),
        ("qwen/qwen-image-2.1-pro", "1:1"),
        ("google/gemini-nano-banana-2.1", "1K"),
    ):
        body = _image_generation_body(
            model,
            "poster",
            size,
            quality="high",
        )
        assert "quality" not in body
        assert "complexity" not in body
