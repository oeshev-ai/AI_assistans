"""Parallel tool tasks must not double-release the parent budget hold."""

from __future__ import annotations

import asyncio

import pytest

from app.infra import cost_tracker as ct
from app.infra.cost_tracker import CostTracker, isolated_budget_reservation


@pytest.mark.asyncio
async def test_parallel_isolated_reservations_keep_parent_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_DAILY_BUDGET_USD", "10")
    monkeypatch.setenv("LLM_USER_DAILY_BUDGET_USD", "0")
    tracker = CostTracker()
    # Point the module singleton used by isolated_budget_reservation at this tracker.
    original = ct.cost_tracker
    ct.cost_tracker = tracker
    token = ct._budget_holds.set(())
    try:
        await tracker.check_budget(1, estimated=1.0)
        assert ct._reserved_amount() == pytest.approx(1.0)

        async def child_spend() -> None:
            async with isolated_budget_reservation():
                await tracker.check_budget(1, estimated=0.5)
                await tracker.record(kind="stt", amount_usd=0.5, user_id=1)

        await asyncio.gather(child_spend(), child_spend())

        assert ct._reserved_amount() == pytest.approx(1.0)
        day = tracker._day_key()
        # Parent hold 1.0 + two child spends 0.5 each.
        assert await tracker._get(f"cost:day:{day}") == pytest.approx(2.0)
    finally:
        ct._budget_holds.reset(token)
        ct.cost_tracker = original
        await tracker.release_reservation(1)


@pytest.mark.asyncio
async def test_unisolated_parallel_record_corrupts_parent_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without isolation children inherit the parent hold ContextVar copy.

    Partial-burn settle no longer wipes the parent Redis hold twice, but the
    parent context still cannot see child burns — ``isolated_budget_reservation``
    remains required for correct per-task accounting.
    """
    monkeypatch.setenv("LLM_DAILY_BUDGET_USD", "10")
    monkeypatch.setenv("LLM_USER_DAILY_BUDGET_USD", "0")
    tracker = CostTracker()
    token = ct._budget_holds.set(())
    try:
        await tracker.check_budget(1, estimated=1.0)

        async def child_spend() -> None:
            # Inherit parent ContextVar copy (asyncio Task semantics).
            await tracker.check_budget(1, estimated=0.5)
            await tracker.record(kind="stt", amount_usd=0.5, user_id=1)

        await asyncio.gather(child_spend(), child_spend())
        day = tracker._day_key()
        # Parent context still believes the full 1.0 hold is live.
        assert ct._reserved_amount() == pytest.approx(1.0)
        assert await tracker._get(f"cost:day:{day}") == pytest.approx(2.0)
    finally:
        ct._budget_holds.reset(token)
        await tracker.release_reservation(1)
