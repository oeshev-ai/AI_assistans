"""HTTP health-check server for readiness / liveness probes."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

import structlog
from aiohttp import web

from app.infra.metrics import refresh_queue_redis_gauge, refresh_redis_gauge

logger = structlog.get_logger("health")

HealthProbe = Callable[[], Awaitable[None]]
AppSetup = Callable[[web.Application], None]
# Readiness probes hit this often; upstream checks stay fresh within a few seconds.
READY_CACHE_SECONDS = 5.0


async def liveness_check(_request: web.Request) -> web.Response:
    """Process-up probe. Never calls Redis or providers (safe for liveness)."""
    return web.json_response({"status": "alive", "timestamp": time.time()}, status=200)


async def readiness_payload(
    request: web.Request,
    redis_client: Any | None,
    proxy_probe: HealthProbe | None,
    provider_probes: Mapping[str, HealthProbe] | None = None,
) -> tuple[dict[str, Any], int]:
    """Readiness snapshot: Redis, ProxyAPI and extra LLM providers.

    ``components.proxy_api`` is kept for compatibility; ``components.providers``
    lists every provider (``proxyapi`` mirrors ``proxy_api``).
    """
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
            await refresh_redis_gauge(redis_client, role="state")
            await refresh_queue_redis_gauge()
        except Exception as exc:
            logger.warning("health_redis_failed", error=str(exc))
            health["components"]["redis"] = "error"
            health["status"] = "degraded"

    if proxy_probe is None:
        health["components"]["proxy_api"] = "disabled"
    else:
        try:
            await proxy_probe()
            health["components"]["proxy_api"] = "ok"
        except Exception as exc:
            logger.warning("health_proxy_failed", error=str(exc))
            health["components"]["proxy_api"] = "error"
            health["status"] = "degraded"

    providers: dict[str, str] = {}
    if proxy_probe is not None:
        providers["proxyapi"] = health["components"]["proxy_api"]
    for provider_id, probe in sorted((provider_probes or {}).items()):
        try:
            await probe()
            providers[provider_id] = "ok"
        except Exception as exc:
            logger.warning(
                "health_provider_failed", provider=provider_id, error=str(exc)
            )
            providers[provider_id] = "error"
            health["status"] = "degraded"
    if providers:
        health["components"]["providers"] = providers

    if request.app.get("upload_enabled"):
        health["components"]["large_upload"] = "ok"
    else:
        health["components"]["large_upload"] = "disabled"

    status_code = 200 if health["status"] == "healthy" else 503
    return health, status_code


async def health_check(
    request: web.Request,
    redis_client: Any | None,
    proxy_probe: HealthProbe | None,
    provider_probes: Mapping[str, HealthProbe] | None = None,
) -> web.Response:
    """Readiness: Redis, ProxyAPI and extra LLM providers.

    Each call probes upstream. ``GET /ready`` caches that snapshot; use this
    helper when the caller needs a fresh result.
    Use ``GET /live`` for liveness and ``GET /ready`` (or ``/health``) for readiness.
    """
    body, status_code = await readiness_payload(
        request, redis_client, proxy_probe, provider_probes
    )
    return web.json_response(body, status=status_code)


async def start_health_server(
    port: int,
    redis_client: Any | None = None,
    proxy_probe: HealthProbe | None = None,
    setup_app: AppSetup | None = None,
    bind: str | None = None,
    provider_probes: Mapping[str, HealthProbe] | None = None,
    ready_ttl: float = READY_CACHE_SECONDS,
) -> web.AppRunner:
    """Start an aiohttp app with live/ready health endpoints.

    - ``GET /live`` — liveness (process up, no upstream probes)
    - ``GET /ready`` and ``GET /health`` — readiness (Redis + providers)

    Default bind is ``127.0.0.1`` (override with ``HEALTH_BIND``, use
    ``0.0.0.0`` inside Docker so published ports work).
    """
    from app.infra.settings import get_settings

    # Запас под multipart + большие медиа (прокси-upload через наш домен).
    settings = get_settings()
    max_mb = settings.upload_max_mb
    app = web.Application(client_max_size=(max_mb + 8) * 1024 * 1024)
    app["upload_enabled"] = False
    ready_cache: dict[str, Any] = {"until": 0.0, "body": None, "status": 200}
    ready_lock = asyncio.Lock()

    async def _ready(request: web.Request) -> web.Response:
        now = time.monotonic()
        cached = ready_cache["body"]
        if cached is not None and now < ready_cache["until"]:
            return web.json_response(cached, status=ready_cache["status"])
        async with ready_lock:
            now = time.monotonic()
            cached = ready_cache["body"]
            if cached is not None and now < ready_cache["until"]:
                return web.json_response(cached, status=ready_cache["status"])
            body, status = await readiness_payload(
                request, redis_client, proxy_probe, provider_probes
            )
            ready_cache["body"] = body
            ready_cache["status"] = status
            ready_cache["until"] = time.monotonic() + ready_ttl
            return web.json_response(body, status=status)

    app.router.add_get("/live", liveness_check)
    app.router.add_get("/ready", _ready)
    app.router.add_get("/health", _ready)
    if setup_app is not None:
        setup_app(app)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    raw = bind if bind is not None else settings.health_bind
    host = (raw or "127.0.0.1").strip() or "127.0.0.1"
    try:
        site = web.TCPSite(runner, host, port)
        await site.start()
    except Exception:
        await runner.cleanup()
        raise
    logger.info("health_server_started", port=port, bind=host)
    return runner
