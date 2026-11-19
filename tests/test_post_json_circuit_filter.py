"""_post_json must not trip the breaker on client 4xx."""

from __future__ import annotations

import pytest

from app.infra.circuit_breaker import (
    APIUnavailableError,
    AsyncCircuitBreaker,
    CircuitState,
    safe_api_call,
)
from app.llm.errors import AssistAIError, counts_toward_circuit


@pytest.mark.asyncio
async def test_safe_api_call_skips_filtered_failures() -> None:
    breaker = AsyncCircuitBreaker(
        name="filter",
        failure_threshold=2,
        recovery_timeout=60.0,
    )

    async def bad_request() -> None:
        raise AssistAIError("nope", status_code=400, error_code="BAD_REQUEST")

    for _ in range(5):
        with pytest.raises(AssistAIError):
            await safe_api_call(
                breaker,
                bad_request,
                failure_filter=lambda exc: isinstance(exc, AssistAIError)
                and counts_toward_circuit(exc),
            )
    assert breaker.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_safe_api_call_counts_availability_failures() -> None:
    breaker = AsyncCircuitBreaker(
        name="avail",
        failure_threshold=2,
        recovery_timeout=60.0,
    )

    async def unavailable() -> None:
        raise AssistAIError("down", status_code=503, error_code="SERVICE_UNAVAILABLE")

    with pytest.raises(AssistAIError):
        await safe_api_call(
            breaker,
            unavailable,
            failure_filter=lambda exc: isinstance(exc, AssistAIError)
            and counts_toward_circuit(exc),
        )
    with pytest.raises(AssistAIError):
        await safe_api_call(
            breaker,
            unavailable,
            failure_filter=lambda exc: isinstance(exc, AssistAIError)
            and counts_toward_circuit(exc),
        )
    assert breaker.state == CircuitState.OPEN
    with pytest.raises(APIUnavailableError):
        await safe_api_call(
            breaker,
            unavailable,
            user_message="unavailable",
            failure_filter=lambda exc: isinstance(exc, AssistAIError)
            and counts_toward_circuit(exc),
        )
