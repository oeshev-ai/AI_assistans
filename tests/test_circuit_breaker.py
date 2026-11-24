"""Unit tests for the async circuit breaker."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import httpx
import pytest
from respx import MockRouter

from app.infra.circuit_breaker import (
    APIUnavailableError,
    AsyncCircuitBreaker,
    CircuitBreakerError,
    CircuitState,
    get_breaker,
    safe_api_call,
)
from app.llm.assist_ai import AssistAIClient
from app.llm.errors import AssistAIError, counts_toward_circuit


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


def _clock(monkeypatch: pytest.MonkeyPatch) -> dict[str, float]:
    clock = {"t": 100.0}
    monkeypatch.setattr("app.infra.circuit_breaker.time.monotonic", lambda: clock["t"])
    return clock


async def _trip_open(breaker: AsyncCircuitBreaker, clock: dict[str, float]) -> None:
    async def boom() -> None:
        raise RuntimeError("down")

    for _ in range(breaker.failure_threshold):
        with pytest.raises(RuntimeError):
            await breaker.call(boom)
    assert breaker.state == CircuitState.OPEN
    clock["t"] += breaker.recovery_timeout + 1


@pytest.mark.asyncio
async def test_cancelled_half_open_probe_returns_permit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    breaker = AsyncCircuitBreaker(
        name="cancel",
        failure_threshold=1,
        recovery_timeout=10.0,
    )
    clock = _clock(monkeypatch)
    await _trip_open(breaker, clock)

    started = asyncio.Event()

    async def hang() -> None:
        started.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(breaker.call(hang))
    await started.wait()
    assert breaker.state == CircuitState.HALF_OPEN
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert breaker.state == CircuitState.HALF_OPEN

    async def ok() -> str:
        return "back"

    assert await breaker.call(ok) == "back"
    assert breaker.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_generator_close_releases_half_open_permit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    breaker = AsyncCircuitBreaker(
        name="aclose",
        failure_threshold=1,
        recovery_timeout=10.0,
    )
    clock = _clock(monkeypatch)
    await _trip_open(breaker, clock)

    async def chunks() -> AsyncIterator[str]:
        async with breaker.attempt() as probe:
            yield "partial"
            await probe.success()

    agen = chunks()
    assert await anext(agen) == "partial"
    await agen.aclose()
    assert breaker.state == CircuitState.HALF_OPEN
    await breaker.ensure_allows_call()


@pytest.mark.asyncio
async def test_safe_api_call_cancelled_and_filtered_release_permit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    breaker = AsyncCircuitBreaker(
        name="filter-half",
        failure_threshold=1,
        recovery_timeout=10.0,
    )
    clock = _clock(monkeypatch)
    await _trip_open(breaker, clock)

    async def cancel() -> None:
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await safe_api_call(breaker, cancel)
    assert breaker.state == CircuitState.HALF_OPEN

    async def bad_request() -> None:
        raise AssistAIError("nope", status_code=400, error_code="BAD_REQUEST")

    with pytest.raises(AssistAIError):
        await safe_api_call(
            breaker,
            bad_request,
            failure_filter=lambda exc: (
                isinstance(exc, AssistAIError) and counts_toward_circuit(exc)
            ),
        )
    assert breaker.state == CircuitState.HALF_OPEN
    await breaker.ensure_allows_call()


@pytest.mark.asyncio
async def test_abandoned_half_open_probe_is_reclaimed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    breaker = AsyncCircuitBreaker(
        name="abandon",
        failure_threshold=1,
        recovery_timeout=10.0,
    )
    clock = _clock(monkeypatch)
    await _trip_open(breaker, clock)

    generation = await breaker.ensure_allows_call()
    assert breaker.state == CircuitState.HALF_OPEN
    with pytest.raises(CircuitBreakerError):
        await breaker.ensure_allows_call()

    clock["t"] += breaker.recovery_timeout + 1
    await breaker.ensure_allows_call()
    await breaker.release_probe(generation)
    with pytest.raises(CircuitBreakerError):
        await breaker.ensure_allows_call()
    await breaker._on_failure(RuntimeError("late"), generation)
    assert breaker.state == CircuitState.HALF_OPEN


@pytest.mark.asyncio
async def test_slow_half_open_success_still_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    breaker = AsyncCircuitBreaker(
        name="slow",
        failure_threshold=1,
        recovery_timeout=10.0,
    )
    clock = _clock(monkeypatch)
    await _trip_open(breaker, clock)
    generation = await breaker.ensure_allows_call()
    clock["t"] += breaker.recovery_timeout + 100
    await breaker._on_success(generation)
    assert breaker.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_early_stream_close_does_not_stick_half_open(
    assist_client: AssistAIClient,
    respx_mock: MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    breaker = get_breaker("proxyapi")
    clock = _clock(monkeypatch)
    await _trip_open(breaker, clock)

    payload = {
        "choices": [
            {"index": 0, "delta": {"content": "a"}, "finish_reason": None},
        ]
    }
    body = "data: " + json.dumps(payload) + "\n\ndata: [DONE]\n\n"
    respx_mock.post("https://api.proxyapi.ru/v1/chat/completions").mock(
        return_value=httpx.Response(200, text=body)
    )

    messages = {
        "model": "openai/gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
    }
    agen = assist_client._iter_chat_chunks(messages)
    chunk = await anext(agen)
    assert chunk["choices"][0]["delta"]["content"] == "a"
    await agen.aclose()
    assert breaker.state == CircuitState.HALF_OPEN

    async for _chunk in assist_client._iter_chat_chunks(messages):
        pass
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
