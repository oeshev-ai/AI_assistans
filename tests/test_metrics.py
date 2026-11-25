"""Unit tests for Prometheus helpers."""

from __future__ import annotations

import inspect

import pytest

from app.infra import metrics as metrics_mod
from app.infra.metrics import (
    ACTIVE_USERS,
    API_CALLS,
    note_active_user,
    start_metrics_server,
    track_api_call,
)


@pytest.mark.asyncio
async def test_track_api_call_success_and_error() -> None:
    before_ok = API_CALLS.labels(
        service="test", endpoint="ping", status="success"
    )._value.get()
    before_err = API_CALLS.labels(
        service="test", endpoint="ping", status="error"
    )._value.get()

    @track_api_call(service="test", endpoint="ping")
    async def ok() -> str:
        return "ok"

    @track_api_call(service="test", endpoint="ping")
    async def bad() -> str:
        raise ValueError("nope")

    assert await ok() == "ok"
    with pytest.raises(ValueError):
        await bad()

    after_ok = API_CALLS.labels(
        service="test", endpoint="ping", status="success"
    )._value.get()
    after_err = API_CALLS.labels(
        service="test", endpoint="ping", status="error"
    )._value.get()
    assert after_ok == before_ok + 1
    assert after_err == before_err + 1


def test_active_users_gauge() -> None:
    note_active_user(1)
    note_active_user(2)
    note_active_user(1)
    assert ACTIVE_USERS._value.get() >= 2


class _InfoRedis:
    def __init__(self, *, used: int, evicted: int) -> None:
        self.used = used
        self.evicted = evicted

    async def info(self, section: str) -> dict[str, int]:
        if section == "memory":
            return {"used_memory": self.used}
        return {"evicted_keys": self.evicted}


@pytest.mark.asyncio
async def test_eviction_growth_is_logged() -> None:
    from structlog.testing import capture_logs

    from app.infra.metrics import note_evicted_keys, refresh_redis_gauge

    metrics_mod._last_evicted.pop("state", None)
    client = _InfoRedis(used=1000, evicted=4)
    with capture_logs() as logs:
        await refresh_redis_gauge(client, role="state")
        client.evicted = 7
        await refresh_redis_gauge(client, role="state")
    assert metrics_mod.REDIS_SIZE._value.get() == 1000
    assert note_evicted_keys is not None
    evicted = [item for item in logs if item["event"] == "redis_keys_evicted"]
    assert len(evicted) == 1
    assert evicted[0]["delta"] == 3
    assert evicted[0]["role"] == "state"
    metrics_mod._last_evicted.pop("state", None)


def test_metrics_server_defaults_to_localhost_bind() -> None:
    source = inspect.getsource(start_metrics_server)
    assert "METRICS_BIND" in source
    assert "127.0.0.1" in source
    assert callable(metrics_mod.start_metrics_server)
