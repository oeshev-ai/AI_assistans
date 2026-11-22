"""Старый image-id из env резолвится в 2.6-flare, пока флаг не выключен."""

from __future__ import annotations

import pytest
import structlog.testing

from app.llm.assist_ai import AssistAIConfig


def test_edit_model_from_env_aliases_to_flare(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.llm.config.load_dotenv", lambda *_args, **_kwargs: False)
    monkeypatch.setenv("PROXYAPI_API_KEY", "test-key")
    monkeypatch.setenv("POLZA_API_KEY", "test-polza-key")
    monkeypatch.setenv("IMAGE_EDIT_MODEL", "openai/gpt-image-2.5-flare")
    monkeypatch.setenv("IMAGE_LEGACY_ALIASES", "1")
    with structlog.testing.capture_logs() as logs:
        config = AssistAIConfig.from_env()
    assert config.default_image_edit_model == "openai/gpt-image-2.6-flare"
    assert any(
        item.get("event") == "image_model_aliased"
        and item.get("source") == "env"
        and item.get("old") == "openai/gpt-image-2.5-flare"
        and item.get("new") == "openai/gpt-image-2.6-flare"
        for item in logs
    )


def test_alias_flag_off_keeps_legacy_edit_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IMAGE_LEGACY_ALIASES", "0")
    config = AssistAIConfig(
        proxyapi_key="test-key",
        default_image_edit_model="openai/gpt-image-2.5-flare",
    )
    assert config.default_image_edit_model == "openai/gpt-image-2.5-flare"
