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


@pytest.fixture
async def fakeredis_client():
    fakeredis = pytest.importorskip("fakeredis.aioredis")
    client = fakeredis.FakeRedis(decode_responses=True)
    try:
        yield client
    finally:
        await client.aclose()
