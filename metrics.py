"""Prometheus metrics for the Telegram AI assistant bot."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any, ParamSpec, TypeVar

from prometheus_client import Counter, Gauge, Histogram, start_http_server

P = ParamSpec("P")
R = TypeVar("R")

API_CALLS = Counter(
    "api_calls_total",
    "Total API calls to external services",
    ["service", "endpoint", "status"],
)

ERRORS = Counter(
    "errors_total",
    "Total errors by type",
    ["error_type", "module"],
)

LATENCY = Histogram(
    "api_latency_seconds",
    "API request latency in seconds",
    ["service", "endpoint"],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

MESSAGE_LATENCY = Histogram(
    "message_processing_seconds",
    "End-to-end message processing latency",
    ["input_kind"],
    buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
)

ACTIVE_USERS = Gauge(
    "active_users",
    "Number of active users in the last hour",
)

REDIS_SIZE = Gauge(
    "redis_memory_bytes",
    "Redis memory usage in bytes",
)

_metrics_started = False
_active_user_hits: dict[int, float] = {}


def start_metrics_server(port: int = 8080) -> None:
    """Start the Prometheus HTTP scrape endpoint (default ``/metrics``)."""
    global _metrics_started
    if _metrics_started:
        return
    start_http_server(port)
    _metrics_started = True


def note_active_user(user_id: int) -> None:
    """Record activity for the active-users gauge (1-hour sliding window)."""
    now = time.time()
    _active_user_hits[user_id] = now
    cutoff = now - 3600.0
    stale = [uid for uid, ts in _active_user_hits.items() if ts < cutoff]
    for uid in stale:
        del _active_user_hits[uid]
    ACTIVE_USERS.set(len(_active_user_hits))


def track_api_call(
    service: str, endpoint: str
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """Decorator that records success/error counters and latency for async APIs."""

    def decorator(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            started = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                API_CALLS.labels(
                    service=service, endpoint=endpoint, status="success"
                ).inc()
                return result
            except Exception as exc:
                API_CALLS.labels(
                    service=service, endpoint=endpoint, status="error"
                ).inc()
                ERRORS.labels(
                    error_type=type(exc).__name__, module=service
                ).inc()
                raise
            finally:
                LATENCY.labels(service=service, endpoint=endpoint).observe(
                    time.perf_counter() - started
                )

        return wrapper

    return decorator


def observe_message(input_kind: str, seconds: float) -> None:
    """Record how long a user message took to process."""
    MESSAGE_LATENCY.labels(input_kind=input_kind).observe(seconds)


async def refresh_redis_gauge(redis_client: Any | None) -> None:
    """Update ``REDIS_SIZE`` from ``INFO memory`` when a Redis client is available."""
    if redis_client is None:
        REDIS_SIZE.set(0)
        return
    try:
        info = await redis_client.info("memory")
        used = int(info.get("used_memory", 0))
        REDIS_SIZE.set(used)
    except Exception:  # noqa: BLE001, S110 — gauge is best-effort
        # Gauge stays at the last known value; health check reports Redis separately.
        pass
