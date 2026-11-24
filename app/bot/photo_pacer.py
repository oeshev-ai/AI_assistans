"""Пауза между sendPhoto одного чата и один повтор на flood control."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

from aiogram.exceptions import TelegramRetryAfter

_GAP_SECONDS = 1.0

T = TypeVar("T")


class _ChatPhotoPacer:
    __slots__ = ("last", "lock")

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.last = 0.0


_pacers: dict[int, _ChatPhotoPacer] = {}
_pacers_guard = asyncio.Lock()


async def _reserve(chat_id: int) -> None:
    """Ждёт, пока с прошлого sendPhoto этого чата прошла хотя бы секунда."""
    async with _pacers_guard:
        pacer = _pacers.get(chat_id)
        if pacer is None:
            pacer = _ChatPhotoPacer()
            _pacers[chat_id] = pacer
        lock = pacer.lock
    async with lock:
        now = time.monotonic()
        if pacer.last:
            wait = _GAP_SECONDS - (now - pacer.last)
            if wait > 0:
                await asyncio.sleep(wait)
                now = time.monotonic()
        pacer.last = now


async def send_photo_paced(chat_id: int, send: Callable[[], Awaitable[T]]) -> T:
    """Один sendPhoto. На ``TelegramRetryAfter`` ждёт ``retry_after`` и пробует ещё раз."""
    await _reserve(chat_id)
    try:
        return await send()
    except TelegramRetryAfter as exc:
        delay = float(exc.retry_after)
        if delay > 0:
            await asyncio.sleep(delay)
        await _reserve(chat_id)
        return await send()
