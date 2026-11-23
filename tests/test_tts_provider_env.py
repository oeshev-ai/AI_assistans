"""TTS_PROVIDER — поставщик озвучки."""

from __future__ import annotations

import pytest

from app.llm.config import AssistAIConfig


def test_invalid_tts_provider_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROXYAPI_API_KEY", "test-key")
    monkeypatch.setenv("POLZA_API_KEY", "test-polza-key")
    monkeypatch.setenv("TTS_PROVIDER", "banana")
    cfg = AssistAIConfig.from_env()
    assert cfg.tts_provider == "polza"


def test_polza_tts_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROXYAPI_API_KEY", "test-key")
    monkeypatch.setenv("POLZA_API_KEY", "test-polza-key")
    monkeypatch.setenv("TTS_PROVIDER", "polza")
    cfg = AssistAIConfig.from_env()
    assert cfg.tts_provider == "polza"


def test_proxyapi_tts_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROXYAPI_API_KEY", "test-key")
    monkeypatch.setenv("POLZA_API_KEY", "test-polza-key")
    monkeypatch.setenv("TTS_PROVIDER", "proxyapi")
    cfg = AssistAIConfig.from_env()
    assert cfg.tts_provider == "proxyapi"


def test_speechkit_tts_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROXYAPI_API_KEY", "test-key")
    monkeypatch.setenv("POLZA_API_KEY", "test-polza-key")
    monkeypatch.setenv("TTS_PROVIDER", "speechkit")
    monkeypatch.setenv("YC_SPEECHKIT_API_KEY", "yc-key")
    cfg = AssistAIConfig.from_env()
    assert cfg.tts_provider == "speechkit"


def test_tts_voice_and_format_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROXYAPI_API_KEY", "test-key")
    monkeypatch.setenv("POLZA_API_KEY", "test-polza-key")
    monkeypatch.setenv("TTS_PROVIDER", "polza")
    monkeypatch.setenv("TTS_VOICE", "nova")
    monkeypatch.setenv("TTS_FORMAT", "mp3")
    cfg = AssistAIConfig.from_env()
    assert cfg.default_tts_voice == "nova"
    assert cfg.default_tts_format == "mp3"


def test_elevenlabs_tts_model_aliased(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROXYAPI_API_KEY", "test-key")
    monkeypatch.setenv("POLZA_API_KEY", "test-polza-key")
    monkeypatch.setenv("TTS_PROVIDER", "polza")
    monkeypatch.setenv("TTS_MODEL", "elevenlabs/eleven-multilingual-v2")
    cfg = AssistAIConfig.from_env()
    assert cfg.default_tts_model == "openai/gpt-4o-mini-tts"
