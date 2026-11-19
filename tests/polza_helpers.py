"""Хелперы тестов Polza (не фикстуры)."""

from __future__ import annotations

from typing import Any

from app.llm.providers import ProviderConfig


def make_polza_provider(**overrides: Any) -> ProviderConfig:
    """Polza-провайдер с тестовым ключом."""
    fields: dict[str, Any] = {
        "id": "polza",
        "kind": "polza_media",
        "base_url": "https://polza.ai/api",
        "api_key": "test-polza-key",
        "capabilities": ["images", "images_edit"],
        "cost_source": "cost_rub",
        "cost_multiplier": 0.39,
        "upstream_only": ("OpenAI",),
        "allow_upstream_fallbacks": False,
        "max_price_image_rub": 12.0,
    }
    fields.update(overrides)
    return ProviderConfig(**fields)
