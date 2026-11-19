"""Unit tests for LLM cost tracking and budgets."""

from __future__ import annotations

import pytest

from app.infra.cost_tracker import BudgetExceededError, CostTracker


@pytest.mark.asyncio
async def test_budget_blocks_when_exceeded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_DAILY_BUDGET_USD", "0.05")
    tracker = CostTracker()
    tracker.daily_budget = 0.05
    await tracker.record(kind="chat", amount_usd=0.04, user_id=1)
    with pytest.raises(BudgetExceededError):
        await tracker.check_budget(1, estimated=0.02)


@pytest.mark.asyncio
async def test_user_budget_independent(monkeypatch: pytest.MonkeyPatch) -> None:
    tracker = CostTracker()
    tracker.daily_budget = 0
    tracker.user_daily_budget = 0.01
    await tracker.record(kind="image", amount_usd=0.009, user_id=10)
    with pytest.raises(BudgetExceededError):
        await tracker.check_budget(10, estimated=0.005)
    await tracker.check_budget(11, estimated=0.005)


def test_estimate_helpers() -> None:
    tracker = CostTracker()
    assert tracker.estimate_image_cost(2) > tracker.estimate_image_cost(1)
    assert tracker.estimate_stt_cost(60) > 0
    assert tracker.estimate_tts_cost("hello world") > 0
    chat = tracker.estimate_chat_cost(prompt_tokens=1000, completion_tokens=1000)
    assert chat > 0


@pytest.mark.asyncio
async def test_redis_backed_spend(fakeredis_client) -> None:
    tracker = CostTracker()
    tracker.attach_redis(fakeredis_client)
    await tracker.record(kind="chat", amount_usd=0.12, user_id=5)
    day_keys = [k async for k in fakeredis_client.scan_iter(match="cost:day:*")]
    assert day_keys
    assert float(await fakeredis_client.get(day_keys[0])) == pytest.approx(0.12)


@pytest.mark.asyncio
async def test_reserve_then_record_does_not_double_count() -> None:
    tracker = CostTracker()
    tracker.daily_budget = 1.0
    tracker.user_daily_budget = 0
    await tracker.check_budget(1, estimated=0.2)
    await tracker.record(kind="chat", amount_usd=0.15, user_id=1)
    day = tracker._day_key()
    assert await tracker._get(f"cost:day:{day}") == pytest.approx(0.15)


@pytest.mark.asyncio
async def test_release_reservation_clears_failed_hold() -> None:
    tracker = CostTracker()
    tracker.daily_budget = 1.0
    tracker.user_daily_budget = 0.5
    await tracker.check_budget(9, estimated=0.2)
    released = await tracker.release_reservation(9)
    day = tracker._day_key()
    assert released == pytest.approx(0.2)
    assert await tracker._get(f"cost:day:{day}") == pytest.approx(0.0)
    assert await tracker._get(f"cost:user:9:day:{day}") == pytest.approx(0.0)
    await tracker.check_budget(9, estimated=0.4)


@pytest.mark.asyncio
async def test_release_after_record_is_noop() -> None:
    tracker = CostTracker()
    tracker.daily_budget = 1.0
    tracker.user_daily_budget = 0
    await tracker.check_budget(1, estimated=0.2)
    await tracker.record(kind="chat", amount_usd=0.11, user_id=1)
    assert await tracker.release_reservation(1) == 0.0
    day = tracker._day_key()
    assert await tracker._get(f"cost:day:{day}") == pytest.approx(0.11)


@pytest.mark.asyncio
async def test_concurrent_budget_reservations_cap(fakeredis_client) -> None:
    import asyncio

    tracker = CostTracker()
    tracker.attach_redis(fakeredis_client)
    tracker.daily_budget = 0.1
    tracker.user_daily_budget = 0

    async def attempt() -> bool:
        try:
            await tracker.check_budget(1, estimated=0.06)
            return True
        except BudgetExceededError:
            return False

    outcomes = await asyncio.gather(*[attempt() for _ in range(4)])
    assert sum(1 for ok in outcomes if ok) == 1
    assert sum(1 for ok in outcomes if not ok) == 3


@pytest.mark.asyncio
async def test_incr_raises_in_production_on_redis_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")

    class BoomRedis:
        def pipeline(self):
            pipe = type("P", (), {})()

            def incrbyfloat(*_a, **_k):
                return pipe

            def expire(*_a, **_k):
                return pipe

            async def execute():
                raise ConnectionError("redis down incr")

            pipe.incrbyfloat = incrbyfloat  # type: ignore[attr-defined]
            pipe.expire = expire  # type: ignore[attr-defined]
            pipe.execute = execute  # type: ignore[attr-defined]
            return pipe

    tracker = CostTracker()
    tracker.attach_redis(BoomRedis())
    with pytest.raises(ConnectionError):
        await tracker._incr("cost:day:t", 0.5, ttl_seconds=60)
    assert "cost:day:t" not in tracker._memory


@pytest.mark.asyncio
async def test_get_raises_in_production_on_redis_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")

    class BoomRedis:
        async def get(self, _key: str):
            raise ConnectionError("redis down get")

    tracker = CostTracker()
    tracker._memory["cost:day:t"] = 9.9
    tracker.attach_redis(BoomRedis())
    with pytest.raises(ConnectionError):
        await tracker._get("cost:day:t")


@pytest.mark.asyncio
async def test_incr_falls_back_to_memory_outside_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENVIRONMENT", "development")

    class BoomRedis:
        def pipeline(self):
            pipe = type("P", (), {})()

            def incrbyfloat(*_a, **_k):
                return pipe

            def expire(*_a, **_k):
                return pipe

            async def execute():
                raise ConnectionError("redis down incr")

            pipe.incrbyfloat = incrbyfloat  # type: ignore[attr-defined]
            pipe.expire = expire  # type: ignore[attr-defined]
            pipe.execute = execute  # type: ignore[attr-defined]
            return pipe

    tracker = CostTracker()
    tracker.attach_redis(BoomRedis())
    await tracker._incr("cost:day:t", 0.5, ttl_seconds=60)
    assert tracker._memory["cost:day:t"] == pytest.approx(0.5)


def test_require_budgets_fatal_in_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")
    tracker = CostTracker()
    tracker.daily_budget = 0.0
    tracker.user_daily_budget = 0.0
    with pytest.raises(SystemExit):
        tracker.require_llm_budgets_configured()


def test_require_budgets_ok_with_user_cap_in_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")
    tracker = CostTracker()
    tracker.daily_budget = 0.0
    tracker.user_daily_budget = 1.5
    tracker.require_llm_budgets_configured()


def test_require_budgets_ok_unlimited_in_development(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENVIRONMENT", "development")
    tracker = CostTracker()
    tracker.daily_budget = 0.0
    tracker.user_daily_budget = 0.0
    tracker.require_llm_budgets_configured()


@pytest.fixture
async def fakeredis_client():
    fakeredis = pytest.importorskip("fakeredis.aioredis")
    client = fakeredis.FakeRedis(decode_responses=True)
    try:
        yield client
    finally:
        await client.aclose()
