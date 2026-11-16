"""Стоимость картинки растёт с quality, usage важнее таблицы."""

from __future__ import annotations

import pytest

from app.infra.cost_tracker import (
    IMAGE_OUT_USD_PER_1M,
    BudgetExceededError,
    CostTracker,
)

_MODEL = "openai/gpt-image-2.6-flare"


def test_quality_cost_is_monotonic() -> None:
    tracker = CostTracker()
    costs = [
        tracker.estimate_image_cost(1, model=_MODEL, quality=quality, size="1024x1024")
        for quality in ("low", "medium", "high", "xhigh", "max")
    ]
    assert costs == sorted(costs)
    assert costs[0] < costs[-1]


def test_usage_overrides_table() -> None:
    tracker = CostTracker()
    table = tracker.estimate_image_cost(
        1, model=_MODEL, quality="max", size="1024x1024"
    )
    used = tracker.estimate_image_cost(
        1,
        model=_MODEL,
        quality="max",
        size="1024x1024",
        usage={"output_tokens": 100},
    )
    assert used == pytest.approx(100 * IMAGE_OUT_USD_PER_1M / 1_000_000)
    assert used != pytest.approx(table)


def test_non_openai_stays_flat() -> None:
    tracker = CostTracker()
    low = tracker.estimate_image_cost(
        1, model="black-forest-labs/flux-3-image", quality="low", size="1K"
    )
    high = tracker.estimate_image_cost(
        1, model="black-forest-labs/flux-3-image", quality="max", size="1K"
    )
    assert low == high


@pytest.mark.asyncio
async def test_budget_sees_quality_difference() -> None:
    tracker = CostTracker()
    low = tracker.estimate_image_cost(1, model=_MODEL, quality="low", size="1024x1024")
    high = tracker.estimate_image_cost(1, model=_MODEL, quality="max", size="1024x1024")
    tracker.daily_budget = (low + high) / 2
    await tracker.check_budget(1, estimated=low)
    with pytest.raises(BudgetExceededError):
        await tracker.check_budget(1, estimated=high)
