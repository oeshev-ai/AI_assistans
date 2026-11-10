"""Async circuit breaker for external API calls."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from enum import Enum
from functools import wraps
from typing import Any, ParamSpec, TypeVar

import structlog

logger = structlog.get_logger("circuit_breaker")

P = ParamSpec("P")
R = TypeVar("R")


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerError(Exception):
    """Raised when the circuit is open and calls are short-circuited."""


class APIUnavailableError(Exception):
    """User-facing wrapper when a protected service is unavailable."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class AsyncCircuitBreaker:
    """Three-state circuit breaker suitable for asyncio call sites."""

    def __init__(
        self,
        *,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type[BaseException] | tuple[type[BaseException], ...] = (
            Exception,
        ),
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at = 0.0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    async def ensure_allows_call(self) -> None:
        """Raise ``CircuitBreakerError`` if the circuit is currently open."""
        async with self._lock:
            self._maybe_half_open()
            if self._state == CircuitState.OPEN:
                logger.error("circuit_open", circuit=self.name)
                raise CircuitBreakerError(f"Circuit '{self.name}' is open")

    async def call(
        self, func: Callable[P, Awaitable[R]], *args: P.args, **kwargs: P.kwargs
    ) -> R:
        await self.ensure_allows_call()

        try:
            result = await func(*args, **kwargs)
        except self.expected_exception as exc:
            await self._on_failure(exc)
            raise
        else:
            await self._on_success()
            return result

    def _maybe_half_open(self) -> None:
        if self._state != CircuitState.OPEN:
            return
        if time.monotonic() - self._opened_at >= self.recovery_timeout:
            self._state = CircuitState.HALF_OPEN
            logger.info("circuit_half_open", circuit=self.name)

    async def _on_success(self) -> None:
        async with self._lock:
            if self._state != CircuitState.CLOSED:
                logger.info("circuit_closed", circuit=self.name)
            self._failure_count = 0
            self._state = CircuitState.CLOSED

    async def _on_failure(self, exc: BaseException) -> None:
        async with self._lock:
            self._failure_count += 1
            logger.error(
                "circuit_call_failed",
                circuit=self.name,
                error=str(exc),
                failures=self._failure_count,
                state=self._state.value,
            )
            if (
                self._state == CircuitState.HALF_OPEN
                or self._failure_count >= self.failure_threshold
            ):
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                logger.error("circuit_opened", circuit=self.name)


def circuit(
    *,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: type[BaseException] | tuple[type[BaseException], ...] = (
        Exception,
    ),
    name: str = "default",
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """Decorator factory that wraps an async function with a dedicated breaker."""
    breaker = AsyncCircuitBreaker(
        name=name,
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        expected_exception=expected_exception,
    )

    def decorator(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return await breaker.call(func, *args, **kwargs)

        wrapper.circuit_breaker = breaker  # type: ignore[attr-defined]
        return wrapper

    return decorator


llm_breaker = AsyncCircuitBreaker(
    name="llm_api", failure_threshold=5, recovery_timeout=60.0
)
weather_breaker = AsyncCircuitBreaker(
    name="weather_api", failure_threshold=5, recovery_timeout=60.0
)
currency_breaker = AsyncCircuitBreaker(
    name="currency_api", failure_threshold=5, recovery_timeout=60.0
)


async def safe_api_call(
    breaker: AsyncCircuitBreaker,
    func: Callable[..., Awaitable[Any]],
    *args: Any,
    user_message: str = "Сервис временно недоступен",
    **kwargs: Any,
) -> Any:
    """Call ``func`` through ``breaker`` and map open-circuit to a user error."""
    try:
        return await breaker.call(func, *args, **kwargs)
    except CircuitBreakerError as exc:
        raise APIUnavailableError(user_message) from exc
