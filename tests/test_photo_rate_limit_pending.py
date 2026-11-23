"""Rate limit на альбоме не должен оставлять sticky pending."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from app.bot.album import AlbumBuild, IncomingPhoto
from app.bot.handlers.photo import PhotoTurn, deliver_album
from app.storage.pending_media import InMemoryPendingMediaStorage


class _Limiter:
    async def check_text_limit(self, _user_id: int) -> bool:
        return False

    async def get_retry_after(self, _user_id: int, _operation: str) -> float:
        return 42.0


@pytest.mark.asyncio
async def test_deliver_album_clears_pending_on_text_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.bot.photo_flow as flow

    answers: list[str] = []

    class _Message:
        chat = SimpleNamespace(id=11)
        from_user = SimpleNamespace(id=22)

        async def answer(self, text: str) -> None:
            answers.append(text)

    pending = InMemoryPendingMediaStorage()
    turn = PhotoTurn(
        message=_Message(),  # type: ignore[arg-type]
        assistant=SimpleNamespace(),
        topic_storage=SimpleNamespace(),
        assistant_storage=SimpleNamespace(),
        state=SimpleNamespace(),
        pending=pending,
    )

    async def active_profile(_chat_id: int, _storage: Any) -> SimpleNamespace:
        return SimpleNamespace(id="persona-a")

    monkeypatch.setattr(flow.R, "_active_profile", active_profile)
    monkeypatch.setattr(flow, "rate_limiter", _Limiter())

    built = AlbumBuild(
        photos=[
            IncomingPhoto(message_id=1, file_id="f1", size=10, caption="поправь")
        ],
        caption="поправь",
        notice=None,
        error=None,
    )
    await deliver_album(built, turn)

    assert await pending.get(11, "persona-a", 22) is None
    assert answers and "Слишком много запросов" in answers[0]
