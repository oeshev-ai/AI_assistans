"""Per-user cap on in-flight image generation and edits in the bot process.

Семафор живёт в памяти процесса бота: воркер его не берёт, иначе fail-closed
лимитер в процессе без Redis отклонил бы уже оплаченную задачу. Словарь
чистится по счётчику, как лок хода чата.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from app.infra.settings import get_settings


class _UserImageGate:
    """Семафор пользователя и число задач, которые его держат или ждут."""

    __slots__ = ("semaphore", "users")

    def __init__(self, limit: int) -> None:
        self.semaphore = asyncio.Semaphore(limit)
        self.users = 0


_image_gates: dict[int, _UserImageGate] = {}
_image_gates_guard = asyncio.Lock()


@asynccontextmanager
async def image_user_slot(user_id: int | None) -> AsyncIterator[None]:
    """Держит слот на время одного запроса: enqueue и ожидание результата.

    ``user_id is None`` — слот не нужен (нет пользователя для квоты).
    Отмена и исключение отпускают слот в ``finally``.
    """
    if user_id is None:
        yield
        return
    limit = get_settings().image_max_concurrent_per_user
    async with _image_gates_guard:
        gate = _image_gates.get(user_id)
        if gate is None:
            gate = _UserImageGate(limit)
            _image_gates[user_id] = gate
        gate.users += 1
        semaphore = gate.semaphore
    try:
        async with semaphore:
            yield
    finally:
        async with _image_gates_guard:
            gate.users -= 1
            if gate.users == 0 and _image_gates.get(user_id) is gate:
                del _image_gates[user_id]
