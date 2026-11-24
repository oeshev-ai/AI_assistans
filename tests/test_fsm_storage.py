"""FSM Redis keys must carry EXPIRE so volatile-ttl can evict them."""

from __future__ import annotations

import importlib
from typing import Any
from unittest.mock import AsyncMock

import pytest
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.redis import RedisStorage

# app.bot.__init__ binds the name ``main`` to the entrypoint function, so a
# plain ``import app.bot.main`` returns that function, not the module.
bot_main = importlib.import_module("app.bot.main")


@pytest.mark.asyncio
async def test_create_fsm_storage_sets_state_and_data_ttl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = AsyncMock()
    monkeypatch.setattr(bot_main.Redis, "from_url", lambda *_args, **_kwargs: probe)

    captured: dict[str, Any] = {}

    def from_url(_cls: object, url: str, **kwargs: Any) -> AsyncMock:
        captured["url"] = url
        captured["kwargs"] = kwargs
        return AsyncMock()

    monkeypatch.setattr(RedisStorage, "from_url", classmethod(from_url))

    await bot_main.create_fsm_storage("redis://localhost:6379/0")

    assert captured["url"] == "redis://localhost:6379/0"
    assert captured["kwargs"]["state_ttl"] == bot_main._FSM_TTL_SECONDS
    assert captured["kwargs"]["data_ttl"] == bot_main._FSM_TTL_SECONDS
    assert bot_main._FSM_TTL_SECONDS > 0
    probe.ping.assert_awaited()
    probe.aclose.assert_awaited()


@pytest.mark.asyncio
async def test_fsm_keys_receive_expire() -> None:
    fakeredis = pytest.importorskip("fakeredis.aioredis")
    client = fakeredis.FakeRedis(decode_responses=True)
    storage = RedisStorage(
        redis=client,
        state_ttl=bot_main._FSM_TTL_SECONDS,
        data_ttl=bot_main._FSM_TTL_SECONDS,
    )
    key = StorageKey(bot_id=1, chat_id=2, user_id=3)
    await storage.set_state(key, "NewAssistantSG:name")
    await storage.set_data(key, {"draft": "persona"})

    state_ttl = await client.ttl(storage.key_builder.build(key, "state"))
    data_ttl = await client.ttl(storage.key_builder.build(key, "data"))
    assert 0 < state_ttl <= bot_main._FSM_TTL_SECONDS
    assert 0 < data_ttl <= bot_main._FSM_TTL_SECONDS
    await client.aclose()
