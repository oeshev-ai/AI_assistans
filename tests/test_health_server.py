"""Health endpoint must not leak exception details."""

from __future__ import annotations

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from app.infra.health_server import health_check


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
