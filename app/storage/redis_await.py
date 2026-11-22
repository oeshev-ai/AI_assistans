"""Await helpers for redis-py 5 async stubs.

``redis.asyncio`` commands are typed as ``Awaitable[T] | T`` (shared sync/async
stubs). At runtime the async client always returns an awaitable; cast once here
instead of scattering ``type: ignore`` on every set-command call.
"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import TypeVar, cast

_T = TypeVar("_T")


async def await_redis(value: Awaitable[_T] | _T) -> _T:
    return await cast(Awaitable[_T], value)
