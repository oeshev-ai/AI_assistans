"""Health endpoint must not leak exception details."""

from __future__ import annotations

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from app.infra.health_server import health_check, liveness_check


@pytest.mark.asyncio
async def test_liveness_ok_while_readiness_degraded() -> None:
    class FakeRedis:
        async def ping(self) -> bool:
            raise RuntimeError("redis down")

    async def proxy_fail() -> None:
        raise ConnectionError("upstream down")

    app = web.Application()

    async def ready(request: web.Request) -> web.Response:
        return await health_check(request, FakeRedis(), proxy_fail)

    app.router.add_get("/live", liveness_check)
    app.router.add_get("/ready", ready)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        live = await client.get("/live")
        ready_resp = await client.get("/ready")
        assert live.status == 200
        assert (await live.json())["status"] == "alive"
        assert ready_resp.status == 503
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_health_ok() -> None:
    class FakeRedis:
        async def ping(self) -> bool:
            return True

    async def proxy_ok() -> None:
        return None

    app = web.Application()

    async def handler(request: web.Request) -> web.Response:
        return await health_check(request, FakeRedis(), proxy_ok)

    app.router.add_get("/health", handler)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        resp = await client.get("/health")
        assert resp.status == 200
        body = await resp.json()
        assert body["status"] == "healthy"
        assert body["components"]["redis"] == "ok"
        assert body["components"]["proxy_api"] == "ok"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_health_errors_are_opaque() -> None:
    class FakeRedis:
        async def ping(self) -> bool:
            raise RuntimeError("secret-host:6379 auth failed xyz")

    async def proxy_fail() -> None:
        raise ConnectionError("upstream 10.0.0.5 refused")

    app = web.Application()

    async def handler(request: web.Request) -> web.Response:
        return await health_check(request, FakeRedis(), proxy_fail)

    app.router.add_get("/health", handler)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        resp = await client.get("/health")
        assert resp.status == 503
        body = await resp.json()
        assert body["status"] == "degraded"
        assert body["components"]["redis"] == "error"
        assert body["components"]["proxy_api"] == "error"
        raw = await resp.text()
        assert "secret-host" not in raw
        assert "10.0.0.5" not in raw
        assert "xyz" not in raw
    finally:
        await client.close()
