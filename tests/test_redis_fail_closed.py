"""Production Redis create_* helpers must not silently fall back to memory."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.storage.assistant_storage import create_assistant_storage
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
