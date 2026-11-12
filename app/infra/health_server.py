"""HTTP health-check server for readiness / liveness probes."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any

import structlog
from aiohttp import web

from app.infra.metrics import refresh_redis_gauge

logger = structlog.get_logger("health")

HealthProbe = Callable[[], Awaitable[None]]
AppSetup = Callable[[web.Application], None]


async def health_check(
    request: web.Request,
    redis_client: Any | None,
    proxy_probe: HealthProbe | None,
) -> web.Response:
    """Return component status for Redis and ProxyAPI."""
    health: dict[str, Any] = {
        "status": "healthy",
        "timestamp": time.time(),
        "components": {},
    }

    if redis_client is None:
        health["components"]["redis"] = "disabled"
    else:
        try:
            await redis_client.ping()
            health["components"]["redis"] = "ok"
            await refresh_redis_gauge(redis_client)
        except Exception as exc:
            health["components"]["redis"] = f"error: {exc}"
            health["status"] = "degraded"

    if proxy_probe is None:
        health["components"]["proxy_api"] = "disabled"
    else:
        try:
            await proxy_probe()
            health["components"]["proxy_api"] = "ok"
        except Exception as exc:
            health["components"]["proxy_api"] = f"error: {exc}"
            health["status"] = "degraded"

    if request.app.get("upload_enabled"):
        health["components"]["large_upload"] = "ok"
    else:
        health["components"]["large_upload"] = "disabled"

    status_code = 200 if health["status"] == "healthy" else 503
    return web.json_response(health, status=status_code)


async def start_health_server(
    port: int,
    redis_client: Any | None = None,
    proxy_probe: HealthProbe | None = None,
    setup_app: AppSetup | None = None,
) -> web.AppRunner:
    """Start an aiohttp app exposing ``GET /health`` on ``0.0.0.0:port``."""
    import os

    # Запас под multipart + большие медиа (прокси-upload через наш домен).
    max_mb = max(1, int(os.getenv("UPLOAD_MAX_MB", "512") or "512"))
    app = web.Application(client_max_size=(max_mb + 8) * 1024 * 1024)
    app["upload_enabled"] = False

    async def _handler(request: web.Request) -> web.Response:
        return await health_check(request, redis_client, proxy_probe)

    app.router.add_get("/health", _handler)
    if setup_app is not None:
        setup_app(app)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    logger.info("health_server_started", port=port)
    return runner
