"""Health endpoint must not leak exception details."""

from __future__ import annotations

import pytest
from aiohttp import ClientSession, web
from aiohttp.test_utils import TestClient, TestServer

from app.infra.health_server import health_check, liveness_check, start_health_server


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


def _bound_port(runner: web.AppRunner) -> int:
    for site in runner.sites:
        server = getattr(site, "_server", None)
        sockets = getattr(server, "sockets", None) or ()
        for sock in sockets:
            return int(sock.getsockname()[1])
    raise AssertionError("health server has no listening socket")


@pytest.mark.asyncio
async def test_ready_reuses_provider_probe_within_ttl() -> None:
    calls = {"n": 0}

    class FakeRedis:
        async def ping(self) -> bool:
            calls["n"] += 1
            return True

    async def proxy_ok() -> None:
        calls["n"] += 1

    runner = await start_health_server(
        port=0,
        redis_client=FakeRedis(),
        proxy_probe=proxy_ok,
        bind="127.0.0.1",
        ready_ttl=30,
    )
    try:
        port = _bound_port(runner)
        async with ClientSession() as session:
            first = await session.get(f"http://127.0.0.1:{port}/ready")
            second = await session.get(f"http://127.0.0.1:{port}/ready")
            live = await session.get(f"http://127.0.0.1:{port}/live")
            assert first.status == 200
            assert second.status == 200
            assert (await first.json())["status"] == "healthy"
            assert (await second.json())["components"]["proxy_api"] == "ok"
            assert live.status == 200
        assert calls["n"] == 2
    finally:
        await runner.cleanup()
