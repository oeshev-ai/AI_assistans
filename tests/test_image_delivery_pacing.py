"""Пауза между sendPhoto и один повтор на flood control."""

from __future__ import annotations

import time
from collections.abc import Iterator

import pytest
from aiogram.exceptions import TelegramRetryAfter
from aiogram.methods import SendMessage

from app.bot import photo_pacer
from app.bot.image_retry import _send_bytes
from app.bot.photo_pacer import send_photo_paced


@pytest.fixture(autouse=True)
def _clear_pacers() -> Iterator[None]:
    photo_pacer._pacers.clear()
    yield
    photo_pacer._pacers.clear()


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> dict[str, float]:
    state = {"t": 50.0}
    sleeps: list[float] = []

    def monotonic() -> float:
        return state["t"]

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        state["t"] += seconds

    monkeypatch.setattr(photo_pacer.time, "monotonic", monotonic)
    monkeypatch.setattr(photo_pacer.asyncio, "sleep", sleep)
    monkeypatch.setattr("app.bot.runtime.asyncio.sleep", sleep)
    monkeypatch.setattr(time, "monotonic", monotonic)
    state["sleeps"] = sleeps  # type: ignore[assignment]
    return state


def _flood(retry_after: int) -> TelegramRetryAfter:
    return TelegramRetryAfter(
        method=SendMessage(chat_id=1, text="x"),
        message="flood",
        retry_after=retry_after,
    )


@pytest.mark.asyncio
async def test_same_chat_waits_at_least_one_second(clock: dict[str, float]) -> None:
    async def send() -> str:
        return "ok"

    await send_photo_paced(4, send)
    await send_photo_paced(8, send)
    assert clock["sleeps"] == []
    await send_photo_paced(4, send)
    assert clock["sleeps"] == [1.0]


@pytest.mark.asyncio
async def test_retry_after_is_slept_once_and_next_photo_still_sends(
    clock: dict[str, float],
) -> None:
    sent: list[str] = []

    class _Bot:
        async def send_photo(
            self,
            chat_id: int,
            photo: object,
            caption: str | None = None,
            request_timeout: int | None = None,
        ) -> None:
            sent.append(getattr(photo, "filename", ""))
            if len(sent) <= 2:
                raise _flood(3)

        async def send_document(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("document")

    bot = _Bot()
    failed: list[str] = []
    for name in ("a.png", "b.png"):
        try:
            await _send_bytes(bot, 9, b"png", name)  # type: ignore[arg-type]
        except TelegramRetryAfter:
            failed.append(name)
    assert failed == ["a.png"]
    assert sent == ["a.png", "a.png", "b.png"]
    assert 3.0 in clock["sleeps"]


@pytest.mark.asyncio
async def test_long_photo_flood_is_not_waited_out(clock: dict[str, float]) -> None:
    calls = 0

    async def send() -> str:
        nonlocal calls
        calls += 1
        raise _flood(45)

    with pytest.raises(TelegramRetryAfter):
        await send_photo_paced(4, send)
    assert calls == 1
    assert clock["sleeps"] == []
