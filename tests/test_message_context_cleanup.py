"""Ранний выход хода не оставляет медиа и поля лога в задаче."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import structlog

from app.bot import runtime as R
from app.bot.constants import MSG_HISTORY, MSG_PROFILE
from app.media.media_tools import (
    UserMedia,
    get_user_media,
    reset_user_media,
    set_user_media,
)
from app.storage.assistant_storage import AssistantStorageError
from app.storage.topic_history import TopicStorageError
from app.storage.upload_sessions import get_upload_request_context


class _Chat:
    id = 7


class _User:
    id = 11


class _Message:
    def __init__(self) -> None:
        self.chat = _Chat()
        self.from_user = _User()
        self.replies: list[str] = []

    async def answer(self, text: str) -> None:
        self.replies.append(text)


class _Profiles:
    def __init__(self, *, fail: bool) -> None:
        self.fail = fail

    async def get_active_assistant(self, chat_id: int) -> SimpleNamespace:
        if self.fail:
            raise AssistantStorageError("down")
        return SimpleNamespace(id="persona")


class _Topics:
    async def ensure_active(self, chat_id: int, assistant_id: str, name: str) -> str:
        raise TopicStorageError("down")


def _clip() -> UserMedia:
    return UserMedia(
        kind="audio",
        filename="clip.ogg",
        mime_type="audio/ogg",
        data=b"ogg",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("profile_fails", "expected"),
    [(True, MSG_PROFILE), (False, MSG_HISTORY)],
)
async def test_storage_error_clears_turn_context(
    monkeypatch: pytest.MonkeyPatch,
    profile_fails: bool,
    expected: str,
) -> None:
    async def _budget(*_args: object, **_kwargs: object) -> None:
        return None

    async def _release(*_args: object, **_kwargs: object) -> float:
        return 0.0

    monkeypatch.setattr(R.cost_tracker, "check_budget", _budget)
    monkeypatch.setattr(R.cost_tracker, "release_reservation", _release)
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(user_id=1, chat_id=1)
    outer = set_user_media(
        [
            UserMedia(
                kind="image",
                filename="outer.png",
                mime_type="image/png",
                data=b"png",
            )
        ]
    )
    message = _Message()
    try:
        await R._process_user_message_locked(
            message,  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
            _Topics(),  # type: ignore[arg-type]
            _Profiles(fail=profile_fails),  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
            user_text="привет",
            user_media=[_clip()],
        )
        assert message.replies == [expected]
        assert [item.filename for item in get_user_media()] == ["outer.png"]
        assert get_upload_request_context() is None
        assert "user_id" not in structlog.contextvars.get_contextvars()
        assert "chat_id" not in structlog.contextvars.get_contextvars()
    finally:
        reset_user_media(outer)
        structlog.contextvars.clear_contextvars()


@pytest.mark.asyncio
async def test_unexpected_error_releases_budget_and_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    released: list[int] = []

    async def _budget(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("redis after hold")

    async def _release(user_id: int | None = None) -> float:
        released.append(0 if user_id is None else user_id)
        return 0.0

    monkeypatch.setattr(R.cost_tracker, "check_budget", _budget)
    monkeypatch.setattr(R.cost_tracker, "release_reservation", _release)
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(user_id=1, chat_id=1)
    outer = set_user_media(
        [
            UserMedia(
                kind="image",
                filename="outer.png",
                mime_type="image/png",
                data=b"png",
            )
        ]
    )
    message = _Message()
    try:
        with pytest.raises(RuntimeError, match="redis after hold"):
            await R._process_user_message_locked(
                message,  # type: ignore[arg-type]
                None,  # type: ignore[arg-type]
                _Topics(),  # type: ignore[arg-type]
                _Profiles(fail=False),  # type: ignore[arg-type]
                None,  # type: ignore[arg-type]
                user_text="привет",
                user_media=[_clip()],
            )
        assert released == [11]
        assert message.replies == []
        assert [item.filename for item in get_user_media()] == ["outer.png"]
        assert get_upload_request_context() is None
        assert structlog.contextvars.get_contextvars() == {}
    finally:
        reset_user_media(outer)
        structlog.contextvars.clear_contextvars()


@pytest.mark.asyncio
async def test_profile_validation_error_releases_budget_and_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    released: list[int] = []

    async def _budget(*_args: object, **_kwargs: object) -> None:
        return None

    async def _release(user_id: int | None = None) -> float:
        released.append(0 if user_id is None else user_id)
        return 0.0

    class _BrokenProfiles:
        async def get_active_assistant(self, chat_id: int) -> SimpleNamespace:
            raise ValueError("битый профиль")

    monkeypatch.setattr(R.cost_tracker, "check_budget", _budget)
    monkeypatch.setattr(R.cost_tracker, "release_reservation", _release)
    structlog.contextvars.clear_contextvars()
    outer = set_user_media([])
    message = _Message()
    try:
        with pytest.raises(ValueError, match="битый профиль"):
            await R._process_user_message_locked(
                message,  # type: ignore[arg-type]
                None,  # type: ignore[arg-type]
                _Topics(),  # type: ignore[arg-type]
                _BrokenProfiles(),  # type: ignore[arg-type]
                None,  # type: ignore[arg-type]
                user_text="привет",
            )
        assert released == [11]
        assert get_upload_request_context() is None
        assert structlog.contextvars.get_contextvars() == {}
    finally:
        reset_user_media(outer)
        structlog.contextvars.clear_contextvars()


@pytest.mark.asyncio
async def test_vision_profile_error_releases_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    released: list[int] = []

    async def _budget(*_args: object, **_kwargs: object) -> None:
        return None

    async def _release(user_id: int | None = None) -> float:
        released.append(0 if user_id is None else user_id)
        return 0.0

    class _BrokenProfiles:
        async def get_active_assistant(self, chat_id: int) -> SimpleNamespace:
            raise ValueError("битый профиль")

    monkeypatch.setattr(R.cost_tracker, "check_budget", _budget)
    monkeypatch.setattr(R.cost_tracker, "release_reservation", _release)
    outer = set_user_media([])
    message = _Message()
    try:
        with pytest.raises(ValueError, match="битый профиль"):
            await R._process_vision_message_locked(
                message,  # type: ignore[arg-type]
                None,  # type: ignore[arg-type]
                _Topics(),  # type: ignore[arg-type]
                _BrokenProfiles(),  # type: ignore[arg-type]
                None,  # type: ignore[arg-type]
                multimodal_content=[],
                history_placeholder="[фото]",
                user_media=[_clip()],
            )
        assert released == [11]
        assert get_user_media() == []
    finally:
        reset_user_media(outer)
