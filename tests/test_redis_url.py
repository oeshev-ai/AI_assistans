"""REDIS_URL для Compose: спецсимволы пароля не ломают разбор."""

from __future__ import annotations

from urllib.parse import urlparse

import pytest

from app.infra.job_queue import _redis_settings_from_url
from app.infra.redis_url import compose_redis_url


def test_empty_password_has_no_userinfo() -> None:
    assert compose_redis_url("redis", "") == "redis://redis:6379/0"
    assert compose_redis_url("  redis  ", "") == "redis://redis:6379/0"


def test_password_is_percent_encoded_and_roundtrips() -> None:
    secret = "p@ss/w:rd #"
    url = compose_redis_url("redis", secret)
    assert url == "redis://:p%40ss%2Fw%3Ard%20%23@redis:6379/0"
    parsed = urlparse(url)
    assert parsed.hostname == "redis"
    assert "@" not in (parsed.password or "")
    settings = _redis_settings_from_url(url)
    assert settings.password == secret
    assert settings.host == "redis"


def test_worker_settings_unquote_password(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.infra.worker import redis_settings

    monkeypatch.setenv("REDIS_URL", compose_redis_url("redis", "a/b@c", db=1))
    settings = redis_settings()
    assert settings.password == "a/b@c"
    assert settings.database == 1
    assert settings.host == "redis"
