"""Темы диалога: память процесса и Redis через fakeredis."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable
from typing import TypeVar

import pytest
import pytest_asyncio
from fakeredis import FakeAsyncRedis
from redis.exceptions import RedisError, WatchError

from app.storage.topic_history import (
    InMemoryTopicStorage,
    RedisTopicStorage,
    TopicConfig,
    TopicHistoryStorage,
    TopicStorageError,
)

_T = TypeVar("_T")


async def test_append_messages_is_abstract() -> None:
    assert "append_messages" in TopicHistoryStorage.__abstractmethods__


async def _await_redis(value: Awaitable[_T] | _T) -> _T:
    """fakeredis/redis-py: ответ может быть и значением, и awaitable."""
    if isinstance(value, Awaitable):
        return await value
    return value


pytestmark = pytest.mark.asyncio

AID = "standard"
OTHER = "other"


@pytest_asyncio.fixture
async def memory_storage() -> AsyncIterator[InMemoryTopicStorage]:
    storage = InMemoryTopicStorage(max_length=20)
    yield storage
    await storage.close()


@pytest_asyncio.fixture
async def redis_storage() -> AsyncIterator[RedisTopicStorage]:
    client = FakeAsyncRedis(decode_responses=True)
    config = TopicConfig(
        redis_url="redis://localhost:6379/0", topic_ttl_days=30, max_history_length=20
    )
    storage = RedisTopicStorage(config, redis=client)
    yield storage
    await storage.close()
    await client.aclose()


def _pick(
    name: str,
    memory_storage: InMemoryTopicStorage,
    redis_storage: RedisTopicStorage,
) -> TopicHistoryStorage:
    if name == "memory":
        return memory_storage
    return redis_storage


@pytest.mark.parametrize("storage_name", ["memory", "redis"])
async def test_topics_do_not_share_history(
    storage_name: str,
    memory_storage: InMemoryTopicStorage,
    redis_storage: RedisTopicStorage,
) -> None:
    storage = _pick(storage_name, memory_storage, redis_storage)
    general = await storage.create_topic(1, AID, "Общая")
    work = await storage.create_topic(1, AID, "Работа")
    await storage.set_active_topic(1, AID, work)
    await storage.append_message(
        1, AID, general, {"role": "user", "content": "погода"}, 20
    )
    await storage.append_message(1, AID, work, {"role": "user", "content": "отчёт"}, 20)

    assert await storage.get_active_topic(1, AID) == work
    assert [item["content"] for item in await storage.get_history(1, AID, general)] == [
        "погода"
    ]
    assert [item["content"] for item in await storage.get_history(1, AID, work)] == [
        "отчёт"
    ]


@pytest.mark.parametrize("storage_name", ["memory", "redis"])
async def test_clear_topic_keeps_the_other(
    storage_name: str,
    memory_storage: InMemoryTopicStorage,
    redis_storage: RedisTopicStorage,
) -> None:
    storage = _pick(storage_name, memory_storage, redis_storage)
    general = await storage.create_topic(2, AID, "Общая")
    work = await storage.create_topic(2, AID, "Работа")
    await storage.append_message(2, AID, general, {"role": "user", "content": "a"}, 20)
    await storage.append_message(2, AID, work, {"role": "user", "content": "b"}, 20)
    await storage.clear_topic(2, AID, general)

    assert await storage.get_history(2, AID, general) == []
    assert [item["content"] for item in await storage.get_history(2, AID, work)] == [
        "b"
    ]
    assert {item["id"] for item in await storage.list_topics(2, AID)} == {general, work}


@pytest.mark.parametrize("storage_name", ["memory", "redis"])
async def test_system_prompt_preserved_inside_topic(
    storage_name: str,
    memory_storage: InMemoryTopicStorage,
    redis_storage: RedisTopicStorage,
) -> None:
    storage = _pick(storage_name, memory_storage, redis_storage)
    topic_id = await storage.create_topic(3, AID, "Общая")
    await storage.append_message(
        3, AID, topic_id, {"role": "system", "content": "sys"}, 20
    )
    for index in range(30):
        role = "user" if index % 2 == 0 else "assistant"
        await storage.append_message(
            3, AID, topic_id, {"role": role, "content": str(index)}, 20
        )
    messages = await storage.get_history(3, AID, topic_id)
    assert messages[0] == {"role": "system", "content": "sys"}
    assert len(messages) <= 20
    assert messages[1]["role"] != "assistant"


@pytest.mark.parametrize("storage_name", ["memory", "redis"])
async def test_ensure_active_creates_general_once(
    storage_name: str,
    memory_storage: InMemoryTopicStorage,
    redis_storage: RedisTopicStorage,
) -> None:
    storage = _pick(storage_name, memory_storage, redis_storage)
    first = await storage.ensure_active(4, AID, "Общая")
    second = await storage.ensure_active(4, AID, "Общая")
    topics = await storage.list_topics(4, AID)
    assert first == second
    assert topics == [{"id": first, "name": "Общая"}]
    assert await storage.get_active_topic(4, AID) == first


@pytest.mark.parametrize("storage_name", ["memory", "redis"])
async def test_switch_unknown_topic(
    storage_name: str,
    memory_storage: InMemoryTopicStorage,
    redis_storage: RedisTopicStorage,
) -> None:
    storage = _pick(storage_name, memory_storage, redis_storage)
    await storage.create_topic(5, AID, "Общая")
    with pytest.raises(TopicStorageError):
        await storage.set_active_topic(5, AID, "missing")


@pytest.mark.parametrize("storage_name", ["memory", "redis"])
async def test_delete_topic_moves_active_and_drops_history(
    storage_name: str,
    memory_storage: InMemoryTopicStorage,
    redis_storage: RedisTopicStorage,
) -> None:
    storage = _pick(storage_name, memory_storage, redis_storage)
    general = await storage.create_topic(7, AID, "Общая")
    work = await storage.create_topic(7, AID, "Работа")
    await storage.set_active_topic(7, AID, work)
    await storage.append_message(7, AID, work, {"role": "user", "content": "x"}, 20)

    await storage.delete_topic(7, AID, work)

    assert await storage.get_active_topic(7, AID) == general
    assert await storage.list_topics(7, AID) == [{"id": general, "name": "Общая"}]
    assert await storage.get_history(7, AID, work) == []
    if isinstance(storage, RedisTopicStorage):
        assert (
            await storage._redis.get(f"chat:7:asst:{AID}:topic:{work}:history") is None
        )


@pytest.mark.parametrize("storage_name", ["memory", "redis"])
async def test_delete_last_topic_clears_active(
    storage_name: str,
    memory_storage: InMemoryTopicStorage,
    redis_storage: RedisTopicStorage,
) -> None:
    storage = _pick(storage_name, memory_storage, redis_storage)
    only = await storage.create_topic(8, AID, "Одна")
    await storage.set_active_topic(8, AID, only)

    await storage.delete_topic(8, AID, only)

    assert await storage.get_active_topic(8, AID) == ""
    assert await storage.list_topics(8, AID) == []


@pytest.mark.parametrize("storage_name", ["memory", "redis"])
async def test_rename_topic_and_unknown(
    storage_name: str,
    memory_storage: InMemoryTopicStorage,
    redis_storage: RedisTopicStorage,
) -> None:
    storage = _pick(storage_name, memory_storage, redis_storage)
    topic_id = await storage.create_topic(9, AID, "Старое")
    await storage.rename_topic(9, AID, topic_id, "  Новое  ")
    assert await storage.list_topics(9, AID) == [{"id": topic_id, "name": "Новое"}]
    with pytest.raises(TopicStorageError):
        await storage.rename_topic(9, AID, "missing", "Имя")
    with pytest.raises(TopicStorageError):
        await storage.delete_topic(9, AID, "missing")


@pytest.mark.parametrize("storage_name", ["memory", "redis"])
async def test_topic_names_are_unique_per_persona(
    storage_name: str,
    memory_storage: InMemoryTopicStorage,
    redis_storage: RedisTopicStorage,
) -> None:
    storage = _pick(storage_name, memory_storage, redis_storage)
    first = await storage.create_topic(20, AID, "Работа")
    second = await storage.create_topic(20, AID, "  Работа ")
    third = await storage.create_topic(20, AID, "Работа")
    other = await storage.create_topic(20, OTHER, "Работа")
    draft = await storage.create_topic(20, AID, "Черновик")

    names = {item["id"]: item["name"] for item in await storage.list_topics(20, AID)}
    assert names[first] == "Работа"
    assert names[second] == "Работа (2)"
    assert names[third] == "Работа (3)"
    assert names[draft] == "Черновик"
    assert (await storage.list_topics(20, OTHER))[0]["id"] == other
    assert (await storage.list_topics(20, OTHER))[0]["name"] == "Работа"

    await storage.rename_topic(20, AID, first, "Работа")
    await storage.rename_topic(20, AID, draft, "Работа")
    names = {item["id"]: item["name"] for item in await storage.list_topics(20, AID)}
    assert names[first] == "Работа"
    assert names[draft] == "Работа (4)"

    base = "я" * 100
    await storage.create_topic(21, AID, base)
    await storage.create_topic(21, AID, base)
    long_names = [item["name"] for item in await storage.list_topics(21, AID)]
    assert long_names[0] == base
    assert long_names[1].endswith(" (2)")
    assert len(long_names[1]) == 100

    await storage.create_topic(22, AID, "Общая")
    if isinstance(storage, RedisTopicStorage):
        await storage._redis.delete(storage._active_key(22, AID))
    else:
        storage._chats[(22, AID)].active = ""
    await storage.ensure_active(22, AID, "Общая")
    assert [item["name"] for item in await storage.list_topics(22, AID)] == [
        "Общая",
        "Общая (2)",
    ]


async def test_delete_topic_failure_keeps_catalog_and_history(
    redis_storage: RedisTopicStorage,
) -> None:
    """Сбой транзакции не убирает тему из списка и не стирает её историю."""
    topic_id = await redis_storage.create_topic(11, AID, "Работа")
    await redis_storage.append_message(
        11, AID, topic_id, {"role": "user", "content": "x"}, 20
    )
    history_key = f"chat:11:asst:{AID}:topic:{topic_id}:history"
    client = redis_storage._redis
    real_pipeline = client.pipeline

    def pipeline(*args: object, **kwargs: object) -> object:
        pipe = real_pipeline(*args, **kwargs)
        real_execute = pipe.execute

        async def execute(*exec_args: object, **exec_kwargs: object) -> object:
            queued = [part for item in pipe.command_stack for part in item[0]]
            if history_key in queued:
                raise RedisError("down")
            return await real_execute(*exec_args, **exec_kwargs)

        pipe.execute = execute  # type: ignore[method-assign]
        return pipe

    client.pipeline = pipeline  # type: ignore[method-assign]
    with pytest.raises(TopicStorageError):
        await redis_storage.delete_topic(11, AID, topic_id)
    client.pipeline = real_pipeline  # type: ignore[method-assign]

    assert await redis_storage.list_topics(11, AID) == [
        {"id": topic_id, "name": "Работа"}
    ]
    assert await redis_storage.get_history(11, AID, topic_id) == [
        {"role": "user", "content": "x"}
    ]


async def test_topic_ttl_is_set(redis_storage: RedisTopicStorage) -> None:
    topic_id = await redis_storage.create_topic(6, AID, "Общая")
    await redis_storage.set_active_topic(6, AID, topic_id)
    await redis_storage.append_message(
        6, AID, topic_id, {"role": "user", "content": "привет"}, 20
    )
    history_ttl = await redis_storage._redis.ttl(
        f"chat:6:asst:{AID}:topic:{topic_id}:history"
    )
    topics_ttl = await redis_storage._redis.ttl(f"chat:6:asst:{AID}:topics")
    assert 29 * 24 * 60 * 60 < history_ttl <= 30 * 24 * 60 * 60
    assert 29 * 24 * 60 * 60 < topics_ttl <= 30 * 24 * 60 * 60


@pytest.mark.parametrize("storage_name", ["memory", "redis"])
async def test_assistants_do_not_share_topics(
    storage_name: str,
    memory_storage: InMemoryTopicStorage,
    redis_storage: RedisTopicStorage,
) -> None:
    storage = _pick(storage_name, memory_storage, redis_storage)
    own = await storage.create_topic(11, AID, "Своя")
    await storage.create_topic(11, OTHER, "Чужая")
    await storage.set_active_topic(11, AID, own)

    assert [item["name"] for item in await storage.list_topics(11, AID)] == ["Своя"]
    assert [item["name"] for item in await storage.list_topics(11, OTHER)] == ["Чужая"]
    assert await storage.get_active_topic(11, OTHER) == ""


@pytest.mark.parametrize("storage_name", ["memory", "redis"])
async def test_delete_assistant_topics_keeps_the_other(
    storage_name: str,
    memory_storage: InMemoryTopicStorage,
    redis_storage: RedisTopicStorage,
) -> None:
    storage = _pick(storage_name, memory_storage, redis_storage)
    topic_id = await storage.create_topic(12, AID, "Уйдёт")
    kept = await storage.create_topic(12, OTHER, "Останется")
    await storage.append_message(
        12, AID, topic_id, {"role": "user", "content": "x"}, 20
    )
    await storage.delete_assistant_topics(12, AID)

    assert await storage.list_topics(12, AID) == []
    assert await storage.get_history(12, AID, topic_id) == []
    assert await storage.list_topics(12, OTHER) == [{"id": kept, "name": "Останется"}]


async def test_legacy_catalog_moves_to_the_reader(
    redis_storage: RedisTopicStorage,
) -> None:
    """Старый каталог чата забирает первая персона, которая его прочитала."""
    client = redis_storage._redis
    await _await_redis(
        client.rpush(
            "chat:10:topics",
            json.dumps({"id": "abc", "name": "Старая"}, ensure_ascii=False),
        )
    )
    await _await_redis(client.set("chat:10:active", "abc"))
    await _await_redis(
        client.rpush(
            "chat:10:topic:abc:history",
            json.dumps({"role": "user", "content": "привет"}, ensure_ascii=False),
        )
    )

    topics = await redis_storage.list_topics(10, AID)
    assert topics == [{"id": "abc", "name": "Старая"}]
    assert await redis_storage.get_active_topic(10, AID) == "abc"
    assert await redis_storage.get_history(10, AID, "abc") == [
        {"role": "user", "content": "привет"}
    ]
    assert await _await_redis(client.get("chat:10:topics")) is None
    assert await _await_redis(client.lrange("chat:10:topic:abc:history", 0, -1)) == []
    assert await redis_storage.list_topics(10, OTHER) == []


def _count_exists(storage: RedisTopicStorage) -> list[int]:
    """Подменяет EXISTS и копит число вызовов в одноэлементном списке."""
    calls = [0]
    client = storage._redis
    original = client.exists

    async def counted(*keys: object) -> int:
        calls[0] += 1
        return await _await_redis(original(*keys))

    client.exists = counted  # type: ignore[method-assign]
    return calls


async def test_legacy_probe_is_once_per_persona(
    redis_storage: RedisTopicStorage,
) -> None:
    """Повторные чтение и запись той же пары не ходят в EXISTS."""
    calls = _count_exists(redis_storage)

    await redis_storage.list_topics(3, AID)
    assert calls[0] == 2

    await redis_storage.get_active_topic(3, AID)
    await redis_storage.list_topics(3, AID)
    await redis_storage.get_history(3, AID, "none")
    await redis_storage.create_topic(3, AID, "Новая")
    assert calls[0] == 2

    await redis_storage.list_topics(3, OTHER)
    assert calls[0] == 4
    await redis_storage.get_active_topic(3, OTHER)
    assert calls[0] == 4


async def test_legacy_probe_is_not_cached_after_redis_error(
    redis_storage: RedisTopicStorage,
) -> None:
    client = redis_storage._redis
    original = client.exists
    state = {"fail": True, "calls": 0}

    async def flaky(*keys: object) -> int:
        state["calls"] += 1
        if state["fail"]:
            raise RedisError("down")
        return await _await_redis(original(*keys))

    client.exists = flaky  # type: ignore[method-assign]
    with pytest.raises(TopicStorageError):
        await redis_storage.list_topics(4, AID)

    state["fail"] = False
    assert await redis_storage.list_topics(4, AID) == []
    assert state["calls"] == 3
    await redis_storage.list_topics(4, AID)
    assert state["calls"] == 3


async def test_failed_legacy_migration_is_retried(
    redis_storage: RedisTopicStorage,
) -> None:
    client = redis_storage._redis
    await _await_redis(
        client.rpush(
            "chat:7:topics",
            json.dumps({"id": "abc", "name": "Старая"}, ensure_ascii=False),
        )
    )
    real_pipeline = client.pipeline

    def busy_pipeline(*_args: object, **_kwargs: object) -> object:
        class _Pipe:
            async def __aenter__(self) -> _Pipe:
                return self

            async def __aexit__(self, *_exc: object) -> bool:
                return False

            async def watch(self, *_keys: object) -> None:
                raise WatchError("busy")

        return _Pipe()

    client.pipeline = busy_pipeline  # type: ignore[method-assign]
    with pytest.raises(TopicStorageError):
        await redis_storage.list_topics(7, AID)

    client.pipeline = real_pipeline  # type: ignore[method-assign]
    assert await redis_storage.list_topics(7, AID) == [{"id": "abc", "name": "Старая"}]


async def test_deleting_topics_checks_legacy_again(
    redis_storage: RedisTopicStorage,
) -> None:
    await redis_storage.list_topics(8, AID)
    calls = _count_exists(redis_storage)

    await redis_storage.list_topics(8, AID)
    assert calls[0] == 0
    await redis_storage.delete_assistant_topics(8, AID)
    await redis_storage.list_topics(8, AID)
    assert calls[0] == 2


@pytest.mark.parametrize("storage_name", ["memory", "redis"])
async def test_append_messages_writes_a_tool_round_at_once(
    storage_name: str,
    memory_storage: InMemoryTopicStorage,
    redis_storage: RedisTopicStorage,
) -> None:
    storage = _pick(storage_name, memory_storage, redis_storage)
    topic_id = await storage.create_topic(4, AID, "Рисование")
    await storage.append_message(
        4, AID, topic_id, {"role": "user", "content": "нарисуй"}, 20
    )
    updates = 0
    original = storage._update_history if storage_name == "redis" else None

    async def counting(*args: object, **kwargs: object) -> None:
        nonlocal updates
        updates += 1
        assert original is not None
        await original(*args, **kwargs)

    if original is not None:
        storage._update_history = counting  # type: ignore[method-assign]

    await storage.append_messages(
        4,
        AID,
        topic_id,
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call_1", "type": "function"}],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "{}"},
            {"role": "assistant", "content": "готово"},
        ],
        20,
    )

    history = await storage.get_history(4, AID, topic_id)
    assert [item["role"] for item in history] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    if storage_name == "redis":
        assert updates == 1


@pytest.mark.parametrize("storage_name", ["memory", "redis"])
async def test_append_messages_keeps_history_when_a_later_item_is_invalid(
    storage_name: str,
    memory_storage: InMemoryTopicStorage,
    redis_storage: RedisTopicStorage,
) -> None:
    storage = _pick(storage_name, memory_storage, redis_storage)
    topic_id = await storage.create_topic(5, AID, "Общая")
    await storage.append_message(
        5, AID, topic_id, {"role": "user", "content": "было"}, 20
    )
    with pytest.raises(TopicStorageError):
        await storage.append_messages(
            5,
            AID,
            topic_id,
            [
                {"role": "assistant", "content": "не должно сохраниться"},
                {"content": "без role"},
            ],
            20,
        )
    history = await storage.get_history(5, AID, topic_id)
    assert history == [{"role": "user", "content": "было"}]
