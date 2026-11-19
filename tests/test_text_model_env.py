"""PROXYAPI_TEXT_MODEL must drive AssistAIConfig.default_text_model."""

from __future__ import annotations

import pytest

from app.llm.assist_ai import AssistAIConfig


def test_text_model_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.llm.config.load_dotenv", lambda *_args, **_kwargs: False)
    monkeypatch.setenv("PROXYAPI_API_KEY", "test-key")
    monkeypatch.setenv("PROXYAPI_TEXT_MODEL", "openai/gpt-6-luna")
    config = AssistAIConfig.from_env()
    assert config.default_text_model == "openai/gpt-6-luna"


def test_text_model_default_when_env_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.llm.config.load_dotenv", lambda *_args, **_kwargs: False)
    monkeypatch.setenv("PROXYAPI_API_KEY", "test-key")
    monkeypatch.setenv("PROXYAPI_TEXT_MODEL", "  ")
    config = AssistAIConfig.from_env()
    assert config.default_text_model == "deepseek/deepseek-v4.1-flash"
