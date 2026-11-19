"""Несколько фото: OpenAI оставляет файлы, остальные — ошибка или склейка."""

from __future__ import annotations

import io

import pytest

from app.llm.errors import AssistAIError
from app.llm.image_ops import prepare_edit_files


def _png(color: tuple[int, int, int]) -> bytes:
    from PIL import Image

    image = Image.new("RGB", (8, 8), color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_openai_keeps_separate_files() -> None:
    files = [
        (_png((255, 0, 0)), "a.png", "image/png"),
        (_png((0, 0, 255)), "b.png", "image/png"),
    ]
    prepared = prepare_edit_files(files, "openai/gpt-image-2.6-flare")
    assert len(prepared) == 2
    assert prepared[0][1] == "a.png"


def test_unsupported_without_fallback_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOW_COMPOSITE_FALLBACK", "0")
    files = [
        (_png((255, 0, 0)), "a.png", "image/png"),
        (_png((0, 0, 255)), "b.png", "image/png"),
    ]
    with pytest.raises(AssistAIError, match="несколько изображений"):
        prepare_edit_files(files, "black-forest-labs/flux-3-image")


def test_unsupported_with_fallback_composites(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOW_COMPOSITE_FALLBACK", "1")
    files = [
        (_png((255, 0, 0)), "a.png", "image/png"),
        (_png((0, 0, 255)), "b.png", "image/png"),
    ]
    prepared = prepare_edit_files(files, "black-forest-labs/flux-3-image", ["1", "2"])
    assert len(prepared) == 1
    assert prepared[0][1] == "composite.png"
    assert prepared[0][2] == "image/png"
    assert prepared[0][0].startswith(b"\x89PNG")
