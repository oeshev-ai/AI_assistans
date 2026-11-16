"""Патчи входа и одинаковая оценка портрета и ландшафта."""

from __future__ import annotations

from app.infra.cost_tracker import CostTracker, estimate_input_image_tokens

_MODEL = "openai/gpt-image-2.6-flare"


def test_input_tokens_follow_patches_and_cap() -> None:
    assert estimate_input_image_tokens(1024, 1024) == 1024
    assert estimate_input_image_tokens(1536, 1024) == 1536
    assert estimate_input_image_tokens(4032, 3024) == 1536


def test_high_portrait_and_landscape_cost_the_same() -> None:
    tracker = CostTracker()
    portrait = tracker.estimate_image_cost(
        1, model=_MODEL, quality="high", size="1024x1536"
    )
    landscape = tracker.estimate_image_cost(
        1, model=_MODEL, quality="high", size="1536x1024"
    )
    assert portrait == landscape
    assert portrait > 0
