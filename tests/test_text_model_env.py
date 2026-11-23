"""TEXT_MODEL must drive AssistAIConfig.default_text_model."""

from __future__ import annotations

import pytest
import structlog.testing

from app.llm.assist_ai import AssistAIConfig


def test_text_model_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.llm.config.load_dotenv", lambda *_args, **_kwargs: False)
    monkeypatch.setenv("PROXYAPI_API_KEY", "test-key")
    monkeypatch.setenv("POLZA_API_KEY", "test-polza-key")
    monkeypatch.setenv("TEXT_MODEL", "openai/gpt-6-luna")
    config = AssistAIConfig.from_env()
    assert config.default_text_model == "openai/gpt-6-luna"


def test_photo_router_defaults_to_flash_1017(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.llm.config.load_dotenv", lambda *_args, **_kwargs: False)
    monkeypatch.setenv("PROXYAPI_API_KEY", "test-key")
    monkeypatch.setenv("POLZA_API_KEY", "test-polza-key")
    monkeypatch.delenv("PHOTO_ROUTER_MODEL", raising=False)
    config = AssistAIConfig.from_env()
    assert config.photo_router_model == "deepseek/deepseek-v4.1-flash-1017"


def test_text_model_default_when_env_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.llm.config.load_dotenv", lambda *_args, **_kwargs: False)
    monkeypatch.setenv("PROXYAPI_API_KEY", "test-key")
    monkeypatch.setenv("POLZA_API_KEY", "test-polza-key")
    monkeypatch.setenv("TEXT_MODEL", "  ")
    config = AssistAIConfig.from_env()
    assert config.default_text_model == "deepseek/deepseek-v4.1-flash-1017"


def test_legacy_text_model_from_env_aliased(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.llm.config.load_dotenv", lambda *_args, **_kwargs: False)
    monkeypatch.setenv("PROXYAPI_API_KEY", "test-key")
    monkeypatch.setenv("POLZA_API_KEY", "test-polza-key")
    monkeypatch.setenv("TEXT_MODEL", "qwen/qwen3.8-max-0902")
    monkeypatch.setenv("PHOTO_ROUTER_MODEL", "google/gemini-3.5-flash-lite")
    monkeypatch.setenv("TEXT_LEGACY_ALIASES", "1")
    with structlog.testing.capture_logs() as logs:
        config = AssistAIConfig.from_env()
    assert config.default_text_model == "qwen/qwen4-max"
    assert config.photo_router_model == "google/gemini-3.8-flash"
    assert any(
        item.get("event") == "text_model_aliased"
        and item.get("source") == "env"
        and item.get("old") == "qwen/qwen3.8-max-0902"
        for item in logs
    )
