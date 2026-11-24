"""Утилиты Vision: кодирование, multimodal content, PDF."""

from __future__ import annotations

import base64

import pytest

from app.llm import vision
from app.llm.vision import (
    VisionError,
    build_image_content,
    build_multimodal_content,
    encode_image_base64,
    ensure_within_size_limit,
    process_pdf,
)


def test_encode_image_base64_roundtrip() -> None:
    raw = b"\x89PNG-test"
    encoded = encode_image_base64(raw)
    assert base64.b64decode(encoded) == raw
    assert "data:" not in encoded


def test_build_image_content_uses_data_url() -> None:
    payload = build_image_content(b"abc", "image/png", detail="low")
    assert payload["type"] == "image_url"
    assert payload["image_url"]["detail"] == "low"
    assert payload["image_url"]["url"].startswith("data:image/png;base64,")


def test_build_multimodal_content_default_prompt() -> None:
    image = build_image_content(b"x", "image/jpeg")
    content = build_multimodal_content("  ", [image])
    assert content[0]["type"] == "text"
    assert "Опиши" in content[0]["text"]
    assert content[1] is image


def test_build_multimodal_content_keeps_caption() -> None:
    image = build_image_content(b"x", "image/jpeg")
    content = build_multimodal_content("Что на фото?", [image])
    assert content[0] == {"type": "text", "text": "Что на фото?"}


def test_ensure_within_size_limit_rejects_large() -> None:
    with pytest.raises(VisionError, match="слишком большой"):
        ensure_within_size_limit(b"x" * (21 * 1024 * 1024))


def test_process_pdf_renders_pages() -> None:
    fitz = pytest.importorskip("fitz")
    document = fitz.open()
    page = document.new_page(width=200, height=200)
    page.insert_text((20, 40), "vision-test")
    pdf_bytes = document.tobytes()
    document.close()

    pages = process_pdf(pdf_bytes, max_pages=1)
    assert len(pages) == 1
    assert pages[0][:4] == b"\x89PNG"


def test_giant_mediabox_exceeds_pixel_cap() -> None:
    class _Rect:
        width = 20_000
        height = 20_000

    class _Page:
        rect = _Rect()

    area = vision._page_pixel_area(_Page(), dpi=vision._PDF_RENDER_DPI)
    assert area > vision.MAX_PDF_PIXELS


def test_process_pdf_rejects_oversized_page(monkeypatch: pytest.MonkeyPatch) -> None:
    fitz = pytest.importorskip("fitz")
    document = fitz.open()
    page = document.new_page(width=200, height=200)
    page.insert_text((20, 40), "huge")
    pdf_bytes = document.tobytes()
    document.close()

    monkeypatch.setattr(vision, "MAX_PDF_PIXELS", 1)
    with pytest.raises(VisionError, match="слишком большая"):
        process_pdf(pdf_bytes, max_pages=1)
