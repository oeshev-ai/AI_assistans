"""Clients opened during startup are closed if the health server fails to bind."""

from __future__ import annotations

import asyncio
import importlib
from types import SimpleNamespace

import pytest

from app.bot.main import Dispatcher, on_startup


class _Closable:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class _Assistant(_Closable):
    def provider_ids(self) -> list[str]:
        return []

    def get_tools_schema(self) -> list[object]:
        return []


class _Redis:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


async def _idle(*_args: object, **_kwargs: object) -> None:
    await asyncio.Future()


@pytest.mark.asyncio
async def test_health_server_failure_closes_started_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main = importlib.import_module("app.bot.main")

    redis = _Redis()
    created: dict[str, _Closable] = {}
    job_closed = {"ok": False}

    async def _open(_url: str) -> _Redis:
        return redis

    async def _make(cls: type[_Closable], key: str, *_args: object, **_kwargs: object):
        item = cls()
        created[key] = item
        return item

    async def _init(_url: str) -> None:
        return None

    async def _close_jobs() -> None:
        job_closed["ok"] = True

    async def _health(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("health down")

    settings = SimpleNamespace(
        redis_url="redis://localhost/0",
        photo_router_fast_path=False,
        metrics_port=9,
        health_port=9,
        upload_public_base_url="",
    )
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "Redis", _Redis)
    monkeypatch.setattr(main, "_open_shared_redis", _open)
    monkeypatch.setattr(main.rate_limiter, "attach_redis", lambda *_a, **_k: None)
    monkeypatch.setattr(main.cost_tracker, "attach_redis", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "app.bot.album.get_album_hub",
        lambda: SimpleNamespace(attach_redis=lambda *_a, **_k: None),
    )
    monkeypatch.setattr(main.job_queue, "init_job_queue", _init)
    monkeypatch.setattr(main.job_queue, "close_job_queue", _close_jobs)
    monkeypatch.setattr(main.TopicConfig, "from_env", staticmethod(lambda: object()))
    monkeypatch.setattr(main.RedisConfig, "from_env", staticmethod(lambda: object()))
    monkeypatch.setattr(main.AssistAIConfig, "from_env", staticmethod(lambda: object()))
    monkeypatch.setattr(main, "AssistAIClient", _Assistant)
    monkeypatch.setattr(main, "TopicHistoryStorage", _Closable)
    monkeypatch.setattr(main, "AssistantStorage", _Closable)
    monkeypatch.setattr(main, "PendingMediaStorage", _Closable)
    monkeypatch.setattr(
        main,
        "create_topic_storage",
        lambda cfg: _make(_Closable, "topic", cfg),
    )
    monkeypatch.setattr(
        main,
        "create_assistant_storage",
        lambda cfg: _make(_Closable, "profiles", cfg),
    )
    monkeypatch.setattr(
        main,
        "create_pending_media_storage",
        lambda url: _make(_Closable, "pending", url),
    )
    monkeypatch.setattr(main, "get_storage", lambda: None)
    monkeypatch.setattr(main, "start_metrics_server", lambda **_k: None)
    monkeypatch.setattr(main, "start_health_server", _health)
    monkeypatch.setattr(
        "app.storage.image_delivery.open_image_delivery_store",
        lambda *_a, **_k: object(),
    )
    monkeypatch.setattr(
        "app.storage.video_delivery.open_video_delivery_store",
        lambda *_a, **_k: object(),
    )
    monkeypatch.setattr("app.bot.image_retry.image_delivery_loop", _idle)
    monkeypatch.setattr(
        "app.bot.image_retry.bind_image_delivery", lambda *_a, **_k: None
    )
    monkeypatch.setattr("app.bot.video_retry.video_delivery_loop", _idle)
    monkeypatch.setattr(
        "app.bot.video_retry.bind_video_delivery", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        "app.storage.file_delivery.open_file_delivery_store",
        lambda *_a, **_k: object(),
    )
    monkeypatch.setattr("app.bot.file_retry.file_delivery_loop", _idle)
    monkeypatch.setattr("app.bot.file_retry.bind_file_delivery", lambda *_a, **_k: None)

    dispatcher = Dispatcher()
    with pytest.raises(RuntimeError, match="health down"):
        await on_startup(dispatcher, SimpleNamespace())  # type: ignore[arg-type]

    assistant = dispatcher.workflow_data["assistant"]
    assert assistant.closed
    assert created["topic"].closed
    assert created["profiles"].closed
    assert created["pending"].closed
    assert redis.closed
    assert job_closed["ok"]
    image_task = dispatcher.workflow_data["image_delivery_task"]
    video_task = dispatcher.workflow_data["video_delivery_task"]
    file_task = dispatcher.workflow_data["file_delivery_task"]
    assert image_task.cancelled() or image_task.done()
    assert video_task.cancelled() or video_task.done()
    assert file_task.cancelled() or file_task.done()
