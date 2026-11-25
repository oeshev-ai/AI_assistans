"""Правки стрима реже лимита Telegram, индикатор typing не гаснет."""

from __future__ import annotations

import asyncio

import pytest

from app.bot import runtime as R
from app.bot.constants import STREAM_EDIT_INTERVAL, STREAM_TYPING_INTERVAL


def test_stream_pacing_stays_under_telegram_edit_limit() -> None:
    assert 1.0 <= STREAM_EDIT_INTERVAL <= 1.5
    assert STREAM_TYPING_INTERVAL == pytest.approx(4.0)


@pytest.mark.asyncio
async def test_typing_repeats_until_the_turn_ends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(R, "STREAM_TYPING_INTERVAL", 0.05)
    actions: list[tuple[int, str]] = []

    class _Bot:
        async def send_chat_action(self, *, chat_id: int, action: str) -> None:
            actions.append((chat_id, action))

    async with R._typing_while(_Bot(), 7):  # type: ignore[arg-type]
        await asyncio.sleep(0.12)

    assert [action for _chat, action in actions].count("typing") >= 2
    assert {chat for chat, _action in actions} == {7}
    sent = len(actions)
    await asyncio.sleep(0.08)
    assert len(actions) == sent
