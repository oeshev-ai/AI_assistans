"""Не больше IMAGE_MAX_CONCURRENT_PER_USER картинок одного пользователя сразу."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest

from app.infra.image_gate import _image_gates, image_user_slot
from app.infra.settings import get_settings


@pytest.fixture(autouse=True)
def _clear_image_gates() -> Iterator[None]:
    _image_gates.clear()
    yield
    _image_gates.clear()


@pytest.mark.asyncio
async def test_parallel_calls_stay_within_per_user_cap() -> None:
    limit = get_settings().image_max_concurrent_per_user
    entered = 0
    peak = 0
    release = asyncio.Event()
    counter = asyncio.Lock()

    async def work() -> None:
        nonlocal entered, peak
        async with image_user_slot(21):
            async with counter:
                entered += 1
                peak = max(peak, entered)
            await release.wait()
            async with counter:
                entered -= 1

    tasks = [asyncio.create_task(work()) for _ in range(15)]
    for _ in range(200):
        gate = _image_gates.get(21)
        if gate is not None and gate.users == 15:
            break
        await asyncio.sleep(0)
    else:
        pytest.fail("не все вызовы встали в семафор")

    for _ in range(50):
        if entered == limit:
            break
        await asyncio.sleep(0)
    assert entered == limit
    assert peak <= limit
    release.set()
    await asyncio.gather(*tasks)
    assert _image_gates == {}


@pytest.mark.asyncio
async def test_users_do_not_share_a_semaphore() -> None:
    limit = get_settings().image_max_concurrent_per_user
    release = asyncio.Event()
    entered = {1: 0, 2: 0}

    async def work(user_id: int) -> None:
        async with image_user_slot(user_id):
            entered[user_id] += 1
            await release.wait()
            entered[user_id] -= 1

    tasks = [
        asyncio.create_task(work(user_id)) for user_id in (1, 2) for _ in range(limit)
    ]
    for _ in range(200):
        if entered[1] == limit and entered[2] == limit:
            break
        await asyncio.sleep(0)
    else:
        pytest.fail("пользователи не зашли параллельно")
    release.set()
    await asyncio.gather(*tasks)
    assert _image_gates == {}
