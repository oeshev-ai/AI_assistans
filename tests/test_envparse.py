"""Safe env parsers and HTTPS URL guard."""

from __future__ import annotations

import pytest

from app.infra.envparse import ensure_https_url, env_float, env_int


def test_env_float_invalid_uses_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COST_CHAT_INPUT_PER_1K", "not-a-number")
    assert env_float("COST_CHAT_INPUT_PER_1K", 0.00015) == pytest.approx(0.00015)


def test_env_int_invalid_uses_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UPLOAD_MAX_CONCURRENT", "x")
    assert env_int("UPLOAD_MAX_CONCURRENT", 2) == 2


def test_ensure_https_accepts_https() -> None:
    assert (
        ensure_https_url(
            "https://storage.yandexcloud.net",
            name="YC_S3_ENDPOINT",
            default="https://example.com",
        )
        == "https://storage.yandexcloud.net"
    )


def test_ensure_https_fatal_in_production(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")
    with pytest.raises(SystemExit):
        ensure_https_url(
            "http://insecure.example",
            name="SPEECHKIT_URL",
            default="https://tts.example/ok",
        )


def test_ensure_https_allows_http_outside_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENVIRONMENT", "development")
    assert (
        ensure_https_url(
            "http://local.example",
            name="YC_S3_ENDPOINT",
            default="https://fallback.example",
        )
        == "http://local.example"
    )
