"""Tool exception text must not leak paths/URLs to the model."""

from __future__ import annotations

from app.llm.tooling import exception_text


def test_plain_message_kept() -> None:
    assert exception_text(ValueError("bad input")) == "bad input"


def test_url_is_redacted() -> None:
    text = exception_text(RuntimeError("fetch failed https://internal.example/secret"))
    assert "internal.example" not in text
    assert "RuntimeError" in text


def test_windows_path_is_redacted() -> None:
    text = exception_text(OSError(r"cannot open D:\COURSE\AI_assistans\.env"))
    assert ".env" not in text
    assert "OSError" in text
