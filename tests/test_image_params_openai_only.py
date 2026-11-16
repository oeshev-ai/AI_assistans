"""quality и complexity уходят только моделям OpenAI."""

from __future__ import annotations

from app.llm.assist_ai import _image_generation_body


def test_openai_gets_quality_and_complexity() -> None:
    body = _image_generation_body(
        "openai/gpt-image-2.6-flare",
        "poster",
        "1024x1024",
        quality="high",
        complexity="low",
    )
    assert body["quality"] == "high"
    assert body["complexity"] == "low"


def test_other_families_omit_openai_params() -> None:
    for model, size in (
        ("black-forest-labs/flux-3-image", "1K"),
        ("qwen/qwen-image-2.1-pro", "1:1"),
        ("recraft/recraft-v4.1", "1:1"),
    ):
        body = _image_generation_body(
            model,
            "poster",
            size,
            quality="high",
            complexity="high",
        )
        assert "quality" not in body
        assert "complexity" not in body


def test_send_complexity_flag_drops_the_field() -> None:
    body = _image_generation_body(
        "openai/gpt-image-2.6-flare",
        "poster",
        "1024x1024",
        quality="high",
        complexity="low",
        send_complexity=False,
    )
    assert body["quality"] == "high"
    assert "complexity" not in body
