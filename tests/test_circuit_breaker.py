"""Unit tests for the async circuit breaker."""

from __future__ import annotations

import pytest

from app.infra.circuit_breaker import (
    APIUnavailableError,
    AsyncCircuitBreaker,
    CircuitBreakerError,
    CircuitState,
    safe_api_call,
)


@pytest.mark.asyncio
async def test_opens_after_threshold_failures() -> None:
    breaker = AsyncCircuitBreaker(
        name="test",
        failure_threshold=3,
        recovery_timeout=60.0,
    )

    async def boom() -> None:
        raise RuntimeError("down")

    for _ in range(3):
        with pytest.raises(RuntimeError):
            await breaker.call(boom)
    assert breaker.state == CircuitState.OPEN
    with pytest.raises(CircuitBreakerError):
        await breaker.call(boom)


@pytest.mark.asyncio
async def test_safe_api_call_maps_open_circuit() -> None:
    breaker = AsyncCircuitBreaker(
        name="map",
        failure_threshold=1,
        recovery_timeout=60.0,
    )

    async def boom() -> None:
        raise RuntimeError("down")

    with pytest.raises(RuntimeError):
        await breaker.call(boom)

    with pytest.raises(APIUnavailableError, match="unavailable"):
        await safe_api_call(breaker, boom, user_message="unavailable")


@pytest.mark.asyncio
async def test_recovers_after_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    breaker = AsyncCircuitBreaker(
        name="recover",
        failure_threshold=1,
        recovery_timeout=10.0,
    )
    clock = {"t": 100.0}
    monkeypatch.setattr("app.infra.circuit_breaker.time.monotonic", lambda: clock["t"])

    async def boom() -> None:
        raise RuntimeError("down")

    async def ok() -> str:
        return "ok"

    with pytest.raises(RuntimeError):
        await breaker.call(boom)
    assert breaker.state == CircuitState.OPEN

    clock["t"] += 11.0
    assert await breaker.call(ok) == "ok"
    assert breaker.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_half_open_allows_only_one_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    breaker = AsyncCircuitBreaker(
        name="probe",
        failure_threshold=1,
        recovery_timeout=10.0,
    )
    clock = {"t": 100.0}
    monkeypatch.setattr("app.infra.circuit_breaker.time.monotonic", lambda: clock["t"])

    async def boom() -> None:
        raise RuntimeError("down")

    with pytest.raises(RuntimeError):
        await breaker.call(boom)
    assert breaker.state == CircuitState.OPEN
    clock["t"] += 11.0

    await breaker.ensure_allows_call()
    assert breaker.state == CircuitState.HALF_OPEN
    with pytest.raises(CircuitBreakerError):
        await breaker.ensure_allows_call()
    await breaker._on_success()
    assert breaker.state == CircuitState.CLOSED


def test_counts_toward_circuit_skips_client_errors() -> None:
    from app.llm.errors import AssistAIError, counts_toward_circuit

    assert (
        counts_toward_circuit(
            AssistAIError("bad", status_code=400, error_code="BAD_REQUEST")
        )
        is False
    )
    assert (
        counts_toward_circuit(
            AssistAIError("lim", status_code=429, error_code="TOO_MANY_REQUESTS")
        )
        is False
    )
    assert (
        counts_toward_circuit(
            AssistAIError("down", status_code=503, error_code="SERVICE_UNAVAILABLE")
        )
        is True
    )
