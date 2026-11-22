"""Production Redis create_* helpers must not silently fall back to memory."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.bot.main import _open_shared_redis
from app.infra.worker import attach_worker_billing_redis
from app.storage.assistant_storage import create_assistant_storage
from app.storage.history_storage import (
    InMemoryChatHistoryStorage,
    RedisConfig,
    create_storage,
)
from app.storage.pending_media import create_pending_media_storage
from app.storage.topic_history import create_topic_storage


@pytest.mark.asyncio
async def test_pending_media_empty_url_fatal_in_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")
    with pytest.raises(SystemExit):
        await create_pending_media_storage("")


@pytest.mark.asyncio
async def test_topic_empty_url_fatal_in_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")
    config = SimpleNamespace(redis_url="", max_history_length=20)
    with pytest.raises(SystemExit):
        await create_topic_storage(config)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_history_empty_url_fatal_in_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")
    config = RedisConfig(redis_url="")
    with pytest.raises(SystemExit):
        await create_storage(config)


@pytest.mark.asyncio
async def test_history_unreachable_fatal_in_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")
    config = RedisConfig(redis_url="redis://127.0.0.1:1/0")
    with pytest.raises(SystemExit):
        await create_storage(config)


@pytest.mark.asyncio
async def test_history_empty_url_ok_outside_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENVIRONMENT", "development")
    storage = await create_storage(RedisConfig(redis_url=""))
    assert isinstance(storage, InMemoryChatHistoryStorage)
    await storage.close()


@pytest.mark.asyncio
async def test_assistant_empty_url_fatal_in_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")
    config = SimpleNamespace(redis_url="")
    with pytest.raises(SystemExit):
        await create_assistant_storage(config)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_pending_media_empty_url_ok_outside_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENVIRONMENT", "development")
    store = await create_pending_media_storage("")
    await store.close()


@pytest.mark.asyncio
async def test_shared_redis_empty_url_fatal_in_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")
    with pytest.raises(SystemExit):
        await _open_shared_redis("")


@pytest.mark.asyncio
async def test_shared_redis_unreachable_fatal_in_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")
    with pytest.raises(SystemExit):
        await _open_shared_redis("redis://127.0.0.1:1/0")


@pytest.mark.asyncio
async def test_shared_redis_empty_url_ok_outside_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENVIRONMENT", "development")
    assert await _open_shared_redis("") is None


@pytest.mark.asyncio
async def test_worker_billing_redis_empty_fatal_in_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")
    with pytest.raises(SystemExit):
        await attach_worker_billing_redis("")


@pytest.mark.asyncio
async def test_worker_billing_redis_unreachable_fatal_in_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")
    with pytest.raises(SystemExit):
        await attach_worker_billing_redis("redis://127.0.0.1:1/0")


@pytest.mark.asyncio
async def test_worker_billing_redis_empty_ok_outside_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENVIRONMENT", "development")
    assert await attach_worker_billing_redis("") is None
