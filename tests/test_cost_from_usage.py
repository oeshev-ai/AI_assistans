"""Три измеренных payload и доплата за text output."""

from __future__ import annotations

import pytest

from app.infra.cost_tracker import TEXT_OUT_USD_PER_1M, cost_from_usage


def _usage(
    *,
    text_in: int,
    image_in: int,
    image_out: int,
    text_out: int = 0,
) -> dict[str, object]:
    return {
        "input_tokens": text_in + image_in,
        "input_tokens_details": {
            "text_tokens": text_in,
            "image_tokens": image_in,
        },
        "output_tokens": image_out + text_out,
        "output_tokens_details": {
            "image_tokens": image_out,
            "text_tokens": text_out,
        },
    }


def test_measured_payloads_match_proxyapi() -> None:
    generation = cost_from_usage(_usage(text_in=8, image_in=0, image_out=824))
    edit = cost_from_usage(_usage(text_in=25, image_in=1536, image_out=1353))
    poster = cost_from_usage(_usage(text_in=263, image_in=0, image_out=1353))
    assert generation == pytest.approx(0.02476, abs=1e-6)
    assert edit == pytest.approx(0.053003, abs=1e-6)
    assert poster == pytest.approx(0.041905, abs=1e-6)


def test_text_output_adds_ten_dollars_per_million() -> None:
    base = cost_from_usage(_usage(text_in=8, image_in=0, image_out=824))
    with_text = cost_from_usage(
        _usage(text_in=8, image_in=0, image_out=824, text_out=1000)
    )
    assert with_text - base == pytest.approx(1000 * TEXT_OUT_USD_PER_1M / 1_000_000)
    assert with_text - base == pytest.approx(0.01)
