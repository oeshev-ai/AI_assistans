"""Unit tests for LLM cost tracking and budgets."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.infra.cost_tracker import BudgetExceededError, CostTracker


def _utc_days() -> tuple[str, str]:
    """Сегодня и завтра: оба внутри TTL, выселение их не сотрёт."""
    today = datetime.now(UTC).date()
    return today.strftime("%Y%m%d"), (today + timedelta(days=1)).strftime("%Y%m%d")


@pytest.mark.asyncio
async def test_budget_blocks_when_exceeded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_DAILY_BUDGET_USD", "0.05")
    tracker = CostTracker()
    await tracker.record(kind="chat", amount_usd=0.04, user_id=1)
    with pytest.raises(BudgetExceededError):
        await tracker.check_budget(1, estimated=0.02)


@pytest.mark.asyncio
async def test_user_budget_independent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_DAILY_BUDGET_USD", "0")
    monkeypatch.setenv("LLM_USER_DAILY_BUDGET_USD", "0.01")
    tracker = CostTracker()
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


def test_rates_and_budgets_follow_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_DAILY_BUDGET_USD", "1.25")
    monkeypatch.setenv("LLM_USER_DAILY_BUDGET_USD", "0.5")
    monkeypatch.setenv("COST_IMAGE_USD", "0.5")
    tracker = CostTracker()
    assert tracker.daily_budget == pytest.approx(1.25)
    assert tracker.user_daily_budget == pytest.approx(0.5)
    assert tracker.estimate_image_cost(1) == pytest.approx(0.5)
    monkeypatch.setenv("LLM_DAILY_BUDGET_USD", "3")
    monkeypatch.setenv("COST_IMAGE_USD", "0.2")
    assert tracker.daily_budget == pytest.approx(3)
    assert tracker.estimate_image_cost(1) == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_redis_backed_spend(fakeredis_client) -> None:
    tracker = CostTracker()
    tracker.attach_redis(fakeredis_client)
    await tracker.record(kind="chat", amount_usd=0.12, user_id=5)
    day_keys = [k async for k in fakeredis_client.scan_iter(match="cost:day:*")]
    assert day_keys
    assert float(await fakeredis_client.get(day_keys[0])) == pytest.approx(0.12)


@pytest.mark.asyncio
async def test_reserve_then_record_does_not_double_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_DAILY_BUDGET_USD", "1")
    monkeypatch.setenv("LLM_USER_DAILY_BUDGET_USD", "0")
    tracker = CostTracker()
    await tracker.check_budget(1, estimated=0.2)
    await tracker.record(kind="chat", amount_usd=0.15, user_id=1)
    day = tracker._day_key()
    # Hold remaining 0.05 until release; settled spend visible after release.
    assert await tracker._get(f"cost:day:{day}") == pytest.approx(0.2)
    await tracker.release_reservation(1)
    assert await tracker._get(f"cost:day:{day}") == pytest.approx(0.15)


@pytest.mark.asyncio
async def test_multi_round_record_keeps_remaining_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.infra import cost_tracker as ct

    monkeypatch.setenv("LLM_DAILY_BUDGET_USD", "1")
    monkeypatch.setenv("LLM_USER_DAILY_BUDGET_USD", "0")
    tracker = CostTracker()
    token = ct._budget_holds.set(())
    try:
        await tracker.check_budget(1, estimated=0.5)
        await tracker.record(kind="chat", amount_usd=0.2, user_id=1)
        assert ct._reserved_amount() == pytest.approx(0.3)
        await tracker.record(kind="chat", amount_usd=0.15, user_id=1)
        assert ct._reserved_amount() == pytest.approx(0.15)
        day = tracker._day_key()
        # Redis still shows original hold until leftover is released.
        assert await tracker._get(f"cost:day:{day}") == pytest.approx(0.5)
        await tracker.release_reservation(1)
        assert await tracker._get(f"cost:day:{day}") == pytest.approx(0.35)
        assert ct._reserved_amount() == pytest.approx(0.0)
    finally:
        ct._budget_holds.reset(token)


@pytest.mark.asyncio
async def test_settle_pipeline_updates_day_and_user(fakeredis_client) -> None:
    """MULTI/EXEC settle writes both keys in one round-trip (fakeredis-safe)."""
    tracker = CostTracker()
    tracker.attach_redis(fakeredis_client)
    day = tracker._day_key()
    day_key = f"cost:day:{day}"
    user_key = f"cost:user:7:day:{day}"
    await tracker._settle(
        day_key=day_key,
        user_key=user_key,
        day_delta=0.4,
        user_delta=0.25,
        ttl_seconds=86400,
    )
    assert float(await fakeredis_client.get(day_key)) == pytest.approx(0.4)
    assert float(await fakeredis_client.get(user_key)) == pytest.approx(0.25)


@pytest.mark.asyncio
async def test_release_reservation_clears_failed_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_DAILY_BUDGET_USD", "1")
    monkeypatch.setenv("LLM_USER_DAILY_BUDGET_USD", "0.5")
    tracker = CostTracker()
    await tracker.check_budget(9, estimated=0.2)
    released = await tracker.release_reservation(9)
    day = tracker._day_key()
    assert released == pytest.approx(0.2)
    assert await tracker._get(f"cost:day:{day}") == pytest.approx(0.0)
    assert await tracker._get(f"cost:user:9:day:{day}") == pytest.approx(0.0)
    await tracker.check_budget(9, estimated=0.4)


@pytest.mark.asyncio
async def test_release_after_partial_record_clears_remainder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_DAILY_BUDGET_USD", "1")
    monkeypatch.setenv("LLM_USER_DAILY_BUDGET_USD", "0")
    tracker = CostTracker()
    await tracker.check_budget(1, estimated=0.2)
    await tracker.record(kind="chat", amount_usd=0.11, user_id=1)
    assert await tracker.release_reservation(1) == pytest.approx(0.09)
    day = tracker._day_key()
    assert await tracker._get(f"cost:day:{day}") == pytest.approx(0.11)


@pytest.mark.asyncio
async def test_concurrent_budget_reservations_cap(
    fakeredis_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    monkeypatch.setenv("LLM_DAILY_BUDGET_USD", "0.1")
    monkeypatch.setenv("LLM_USER_DAILY_BUDGET_USD", "0")
    tracker = CostTracker()
    tracker.attach_redis(fakeredis_client)

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
    monkeypatch.setenv("LLM_DAILY_BUDGET_USD", "0")
    monkeypatch.setenv("LLM_USER_DAILY_BUDGET_USD", "0")
    tracker = CostTracker()
    with pytest.raises(SystemExit):
        tracker.require_llm_budgets_configured()


def test_require_budgets_ok_with_user_cap_in_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("LLM_DAILY_BUDGET_USD", "0")
    monkeypatch.setenv("LLM_USER_DAILY_BUDGET_USD", "1.5")
    tracker = CostTracker()
    tracker.require_llm_budgets_configured()


def test_require_budgets_ok_unlimited_in_development(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("LLM_DAILY_BUDGET_USD", "0")
    monkeypatch.setenv("LLM_USER_DAILY_BUDGET_USD", "0")
    tracker = CostTracker()
    tracker.require_llm_budgets_configured()


@pytest.mark.asyncio
async def test_record_after_midnight_stays_on_reservation_day(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.infra import cost_tracker as ct

    reserved_day, next_day = _utc_days()
    current = {"day": reserved_day}
    monkeypatch.setattr(CostTracker, "_day_key", staticmethod(lambda: current["day"]))
    monkeypatch.setenv("LLM_DAILY_BUDGET_USD", "1")
    monkeypatch.setenv("LLM_USER_DAILY_BUDGET_USD", "1")
    tracker = CostTracker()
    token = ct._budget_holds.set(())
    try:
        await tracker.check_budget(4, estimated=0.2)
        current["day"] = next_day
        await tracker.record(kind="chat", amount_usd=0.05, user_id=4)
        # Пока остаток резерва жив, ключ дня N остаётся на полной сумме резерва.
        assert await tracker._get(f"cost:day:{reserved_day}") == pytest.approx(0.2)
        assert await tracker._get(f"cost:user:4:day:{reserved_day}") == pytest.approx(
            0.2
        )
        assert await tracker._get(f"cost:day:{next_day}") == pytest.approx(0.0)
        assert await tracker._get(f"cost:user:4:day:{next_day}") == pytest.approx(0.0)
        assert ct._reserved_amount() == pytest.approx(0.15)
        released = await tracker.release_reservation(4)
        assert released == pytest.approx(0.15)
        assert await tracker._get(f"cost:day:{reserved_day}") == pytest.approx(0.05)
        assert await tracker._get(f"cost:user:4:day:{reserved_day}") == pytest.approx(
            0.05
        )
        assert await tracker._get(f"cost:day:{next_day}") == pytest.approx(0.0)
        assert await tracker._get(f"cost:user:4:day:{next_day}") == pytest.approx(0.0)
        assert ct._reserved_amount() == pytest.approx(0.0)
    finally:
        ct._budget_holds.reset(token)


@pytest.mark.asyncio
async def test_overage_after_midnight_charges_reservation_day(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.infra import cost_tracker as ct

    reserved_day, next_day = _utc_days()
    current = {"day": reserved_day}
    monkeypatch.setattr(CostTracker, "_day_key", staticmethod(lambda: current["day"]))
    monkeypatch.setenv("LLM_DAILY_BUDGET_USD", "1")
    monkeypatch.setenv("LLM_USER_DAILY_BUDGET_USD", "1")
    tracker = CostTracker()
    token = ct._budget_holds.set(())
    try:
        await tracker.check_budget(4, estimated=0.2)
        current["day"] = next_day
        await tracker.record(kind="chat", amount_usd=0.35, user_id=4)
        assert await tracker._get(f"cost:day:{reserved_day}") == pytest.approx(0.35)
        assert await tracker._get(f"cost:user:4:day:{reserved_day}") == pytest.approx(
            0.35
        )
        assert await tracker._get(f"cost:day:{next_day}") == pytest.approx(0.0)
        assert ct._reserved_amount() == pytest.approx(0.0)
    finally:
        ct._budget_holds.reset(token)


@pytest.mark.asyncio
async def test_release_after_midnight_does_not_go_negative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.infra import cost_tracker as ct

    reserved_day, next_day = _utc_days()
    current = {"day": reserved_day}
    monkeypatch.setattr(CostTracker, "_day_key", staticmethod(lambda: current["day"]))
    monkeypatch.setenv("LLM_DAILY_BUDGET_USD", "1")
    monkeypatch.setenv("LLM_USER_DAILY_BUDGET_USD", "1")
    tracker = CostTracker()
    token = ct._budget_holds.set(())
    try:
        await tracker.check_budget(4, estimated=0.2)
        current["day"] = next_day
        assert await tracker.release_reservation(4) == pytest.approx(0.2)
        assert await tracker._get(f"cost:day:{reserved_day}") == pytest.approx(0.0)
        assert await tracker._get(f"cost:user:4:day:{reserved_day}") == pytest.approx(
            0.0
        )
        assert await tracker._get(f"cost:day:{next_day}") == pytest.approx(0.0)
        assert await tracker._get(f"cost:user:4:day:{next_day}") == pytest.approx(0.0)
    finally:
        ct._budget_holds.reset(token)


@pytest.mark.asyncio
async def test_second_reserve_after_midnight_keeps_both_days(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.infra import cost_tracker as ct

    reserved_day, next_day = _utc_days()
    current = {"day": reserved_day}
    monkeypatch.setattr(CostTracker, "_day_key", staticmethod(lambda: current["day"]))
    monkeypatch.setenv("LLM_DAILY_BUDGET_USD", "1")
    monkeypatch.setenv("LLM_USER_DAILY_BUDGET_USD", "1")
    tracker = CostTracker()
    token = ct._budget_holds.set(())
    try:
        await tracker.check_budget(4, estimated=0.2)
        current["day"] = next_day
        await tracker.check_budget(4, estimated=0.3)
        await tracker.record(kind="chat", amount_usd=0.4, user_id=4)
        assert await tracker._get(f"cost:day:{reserved_day}") == pytest.approx(0.4)
        assert await tracker._get(f"cost:user:4:day:{reserved_day}") == pytest.approx(
            0.4
        )
        assert await tracker._get(f"cost:day:{next_day}") == pytest.approx(0.1)
        assert await tracker._get(f"cost:user:4:day:{next_day}") == pytest.approx(0.1)
        assert ct._reserved_amount() == pytest.approx(0.1)
        await tracker.release_reservation(4)
        assert await tracker._get(f"cost:day:{reserved_day}") == pytest.approx(0.4)
        assert await tracker._get(f"cost:day:{next_day}") == pytest.approx(0.0)
        assert await tracker._get(f"cost:user:4:day:{next_day}") == pytest.approx(0.0)
    finally:
        ct._budget_holds.reset(token)


@pytest.fixture
async def fakeredis_client():
    fakeredis = pytest.importorskip("fakeredis.aioredis")
    client = fakeredis.FakeRedis(decode_responses=True)
    try:
        yield client
    finally:
        await client.aclose()
