"""Лок хода чата живёт, пока есть активный или ожидающий ход."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest

from app.bot import runtime


@pytest.fixture(autouse=True)
def _clear_chat_turn_locks() -> Iterator[None]:
    runtime._chat_turn_locks.clear()
    yield
    runtime._chat_turn_locks.clear()


@pytest.mark.asyncio
async def test_idle_chat_drops_its_lock() -> None:
    async with runtime._chat_turn(7):
        assert 7 in runtime._chat_turn_locks
    assert runtime._chat_turn_locks == {}


@pytest.mark.asyncio
async def test_overlapping_turns_share_one_lock_then_drop_it() -> None:
    order: list[str] = []
    started = asyncio.Event()
    release = asyncio.Event()

    async def first() -> None:
        async with runtime._chat_turn(3):
            order.append("first-in")
            started.set()
            await release.wait()
            order.append("first-out")

    async def second() -> None:
        await started.wait()
        async with runtime._chat_turn(3):
            order.append("second-in")

    held = asyncio.create_task(first())
    await started.wait()
    waiting = asyncio.create_task(second())
    for _ in range(50):
        gate = runtime._chat_turn_locks.get(3)
        if gate is not None and gate.users == 2:
            break
        await asyncio.sleep(0)
    else:
        pytest.fail("второй ход не встал в очередь")

    assert len(runtime._chat_turn_locks) == 1
    release.set()
    await held
    await waiting
    assert order == ["first-in", "first-out", "second-in"]
    assert runtime._chat_turn_locks == {}


@pytest.mark.parametrize("entry", ["_edit", "_generate", "_generate_video"])
async def test_photo_branch_holds_the_chat_lock(
    entry: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from app.bot import photo_flow

    order: list[str] = []
    started = asyncio.Event()
    release = asyncio.Event()

    async def body(*_args: object, **_kwargs: object) -> None:
        order.append("photo-in")
        started.set()
        await release.wait()
        order.append("photo-out")

    monkeypatch.setattr(photo_flow, f"{entry}_locked", body)
    turn = SimpleNamespace(message=SimpleNamespace(chat=SimpleNamespace(id=9)))

    async def text() -> None:
        async with runtime._chat_turn(9):
            order.append("text-in")

    photo = asyncio.create_task(
        getattr(photo_flow, entry)(turn, None, None, None, None, "topic")
    )
    await started.wait()
    waiting = asyncio.create_task(text())
    for _ in range(50):
        gate = runtime._chat_turn_locks.get(9)
        if gate is not None and gate.users == 2:
            break
        await asyncio.sleep(0)
    else:
        pytest.fail("текстовый ход не встал в очередь за фото")

    assert order == ["photo-in"]
    release.set()
    await photo
    await waiting
    assert order == ["photo-in", "photo-out", "text-in"]


@pytest.mark.asyncio
async def test_text_context_reads_while_the_chat_lock_is_held() -> None:
    from app.bot import photo_flow

    held = asyncio.Event()
    release = asyncio.Event()

    async def holder() -> None:
        async with runtime._chat_turn(9):
            held.set()
            await release.wait()

    task = asyncio.create_task(holder())
    await held.wait()

    class _Storage:
        async def get_history(
            self, chat_id: int, assistant_id: str, topic_id: str
        ) -> list[dict[str, str]]:
            return [{"role": "user", "content": "сделай ярче"}, {"role": "tool"}]

    try:
        texts = await asyncio.wait_for(
            photo_flow._text_context(_Storage(), 9, "persona", "topic"),
            timeout=1,
        )
        gate = runtime._chat_turn_locks.get(9)
        assert gate is not None and gate.users == 1
    finally:
        release.set()
        await task

    assert texts == [{"role": "user", "content": "сделай ярче"}]
    assert runtime._chat_turn_locks == {}
