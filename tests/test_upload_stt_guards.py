"""Guards for large-upload STT: budget estimate and concurrency helpers."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import app.bot.runtime as runtime
from app.infra.cost_tracker import BudgetExceededError, cost_tracker
from app.infra.rate_limiter import rate_limiter
from app.storage.upload_sessions import UploadSession


def test_estimate_upload_stt_cost_uses_file_size() -> None:
    with_size = runtime._estimate_upload_stt_cost(file_size=400_000, max_upload_mb=512)
    fallback = runtime._estimate_upload_stt_cost(file_size=None, max_upload_mb=1)
    assert with_size == pytest.approx(cost_tracker.estimate_stt_cost(100.0))
    assert fallback == pytest.approx(
        cost_tracker.estimate_stt_cost(max(15.0, 1024 * 1024 / 4000.0))
    )


@pytest.mark.asyncio
async def test_upload_process_respects_media_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime._upload_process_sem = asyncio.Semaphore(2)
    session = UploadSession(
        upload_id="u1",
        token="t",
        chat_id=10,
        user_id=42,
        purpose="transcribe",
        media_kind="audio",
        status="uploaded",
        created_at=0.0,
        expires_at=9999.0,
        object_key="uploads/u1.bin",
        filename="a.mp3",
        mime_type="audio/mpeg",
        file_size=12_000,
    )
    bot: Any = SimpleNamespace(send_message=AsyncMock())
    store: Any = SimpleNamespace(update=AsyncMock())
    storage: Any = SimpleNamespace(
        config=SimpleNamespace(max_upload_mb=64),
        download_bytes=AsyncMock(),
        delete_object=AsyncMock(),
    )
    monkeypatch.setattr(
        rate_limiter, "check_media_limit", AsyncMock(return_value=False)
    )
    monkeypatch.setattr(rate_limiter, "get_retry_after", AsyncMock(return_value=30.0))
    check_budget = AsyncMock()
    monkeypatch.setattr(cost_tracker, "check_budget", check_budget)

    await runtime._process_s3_upload_session(
        bot=bot,
        assistant=MagicMock(),
        topic_storage=MagicMock(),
        assistant_storage=MagicMock(),
        store=store,
        storage=storage,
        session=session,
    )

    assert session.status == "failed"
    assert session.error == "rate_limited"
    check_budget.assert_not_called()
    storage.download_bytes.assert_not_called()
    storage.delete_object.assert_awaited_once_with("uploads/u1.bin")
    bot.send_message.assert_awaited()


@pytest.mark.asyncio
async def test_upload_process_respects_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime._upload_process_sem = asyncio.Semaphore(2)
    session = UploadSession(
        upload_id="u2",
        token="t",
        chat_id=10,
        user_id=7,
        purpose="transcribe",
        media_kind="audio",
        status="uploaded",
        created_at=0.0,
        expires_at=9999.0,
        object_key="uploads/u2.bin",
        filename="a.mp3",
        mime_type="audio/mpeg",
        file_size=12_000,
    )
    bot: Any = SimpleNamespace(send_message=AsyncMock())
    store: Any = SimpleNamespace(update=AsyncMock())
    storage: Any = SimpleNamespace(
        config=SimpleNamespace(max_upload_mb=64),
        download_bytes=AsyncMock(),
        delete_object=AsyncMock(),
    )
    monkeypatch.setattr(rate_limiter, "check_media_limit", AsyncMock(return_value=True))
    monkeypatch.setattr(
        cost_tracker,
        "check_budget",
        AsyncMock(side_effect=BudgetExceededError("budget")),
    )

    await runtime._process_s3_upload_session(
        bot=bot,
        assistant=MagicMock(),
        topic_storage=MagicMock(),
        assistant_storage=MagicMock(),
        store=store,
        storage=storage,
        session=session,
    )

    assert session.status == "failed"
    assert session.error == "budget_exceeded"
    storage.download_bytes.assert_not_called()
    storage.delete_object.assert_awaited_once()
