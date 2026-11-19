"""Invalid TTS_PROVIDER falls back to proxyapi with a warning path."""

from __future__ import annotations

import pytest

from app.llm.config import AssistAIConfig


def test_invalid_tts_provider_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROXYAPI_API_KEY", "test-key")
    monkeypatch.setenv("TTS_PROVIDER", "banana")
    cfg = AssistAIConfig.from_env()
    assert cfg.tts_provider == "proxyapi"


def test_speechkit_tts_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROXYAPI_API_KEY", "test-key")
    monkeypatch.setenv("TTS_PROVIDER", "speechkit")
    monkeypatch.setenv("YC_SPEECHKIT_API_KEY", "yc-key")
    cfg = AssistAIConfig.from_env()
    assert cfg.tts_provider == "speechkit"
