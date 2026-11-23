"""generation_ids must be task-local, not shared on the client instance."""

from __future__ import annotations

import asyncio

import pytest

from app.llm.image_ops import _get_generation_ids, _set_generation_ids


@pytest.mark.asyncio
async def test_generation_ids_do_not_cross_tasks() -> None:
    async def worker(tag: str) -> list[str | None]:
        _set_generation_ids([tag])
        await asyncio.sleep(0.01)
        return _get_generation_ids()

    left, right = await asyncio.gather(worker("A"), worker("B"))
    assert left == ["A"]
    assert right == ["B"]


def test_http_base_url_rejected_in_production(monkeypatch: pytest.MonkeyPatch) -> None:
    from pydantic import ValidationError

    from app.llm.providers import ProviderConfig

    monkeypatch.setenv("ENVIRONMENT", "production")
    with pytest.raises(ValidationError):
        ProviderConfig(
            id="bad",
            kind="openai",
            base_url="http://api.evil.example/v1",
            api_key="k",
            capabilities=frozenset({"chat"}),
        )


def test_http_localhost_allowed_in_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.llm.providers import ProviderConfig

    monkeypatch.setenv("ENVIRONMENT", "production")
    cfg = ProviderConfig(
        id="local",
        kind="openai",
        base_url="http://127.0.0.1:8080/v1",
        api_key="k",
        capabilities=frozenset({"chat"}),
    )
    assert cfg.base_url.startswith("http://127.0.0.1")
