"""Guards for large-upload STT: budget estimate and concurrency helpers."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import app.bot.runtime as runtime
from app.infra import cost_tracker as cost_tracker_mod
from app.infra.cost_tracker import BudgetExceededError, cost_tracker
from app.infra.rate_limiter import rate_limiter
from app.llm.errors import AssistAIError
from app.llm.media_api import TranscriptionResult
from app.storage.upload_sessions import UploadSession


def test_upload_stt_floor_ignores_file_size() -> None:
    floor = runtime._upload_stt_floor_cost()
    size_based = cost_tracker.estimate_stt_cost(300 * 1024 * 1024 / 4000.0)
    assert floor == pytest.approx(cost_tracker.estimate_stt_cost(15.0))
    assert floor < 1.5 < size_based


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


@pytest.mark.asyncio
async def test_upload_process_releases_unused_budget_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Фактическая STT-цена ниже малого резерва: остаток снимается."""
    runtime._upload_process_sem = asyncio.Semaphore(2)
    user_id = 424242
    file_size = 300 * 1024 * 1024
    session = UploadSession(
        upload_id="u3",
        token="t",
        chat_id=10,
        user_id=user_id,
        purpose="transcribe",
        media_kind="audio",
        status="uploaded",
        created_at=0.0,
        expires_at=9999.0,
        object_key="uploads/u3.bin",
        filename="a.mp3",
        mime_type="audio/mpeg",
        file_size=file_size,
    )
    actual = 0.001
    assert runtime._upload_stt_floor_cost() > actual

    bot: Any = SimpleNamespace(
        send_message=AsyncMock(return_value=SimpleNamespace(delete=AsyncMock()))
    )
    store: Any = SimpleNamespace(update=AsyncMock())
    storage: Any = SimpleNamespace(
        config=SimpleNamespace(max_upload_mb=64),
        download_bytes=AsyncMock(return_value=b"audio"),
        delete_object=AsyncMock(),
    )
    monkeypatch.setattr(rate_limiter, "check_media_limit", AsyncMock(return_value=True))
    monkeypatch.setattr(
        "app.media.file_magic.mime_compatible", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(
        runtime,
        "prepare_audio_for_stt",
        AsyncMock(return_value=[(b"audio", "a.mp3")]),
    )
    monkeypatch.setattr(
        "app.media.media_utils.audio_duration_seconds",
        AsyncMock(return_value=10.0),
    )
    monkeypatch.setattr(
        runtime.job_queue,
        "transcribe_audio",
        AsyncMock(
            return_value=TranscriptionResult(
                text="",
                cost_usd=actual,
                cost_rub=0.08,
                cost_source="cost_rub",
                provider_id="proxyapi",
                model="whisper-1",
            )
        ),
    )

    monkeypatch.setenv("LLM_DAILY_BUDGET_USD", "50")
    monkeypatch.setenv("LLM_USER_DAILY_BUDGET_USD", "50")
    token = cost_tracker_mod._budget_holds.set(())
    day = cost_tracker._day_key()
    user_key = f"cost:user:{user_id}:day:{day}"
    day_key = f"cost:day:{day}"
    before_user = await cost_tracker._get(user_key)
    before_day = await cost_tracker._get(day_key)
    try:
        await runtime._process_s3_upload_session(
            bot=bot,
            assistant=MagicMock(),
            topic_storage=MagicMock(),
            assistant_storage=MagicMock(),
            store=store,
            storage=storage,
            session=session,
        )
        assert session.status == "done"
        assert cost_tracker_mod._reserved_amount() == pytest.approx(0.0)
        assert await cost_tracker._get(user_key) - before_user == pytest.approx(actual)
        assert await cost_tracker._get(day_key) - before_day == pytest.approx(actual)
    finally:
        await cost_tracker.release_reservation(user_id)
        cost_tracker._memory[user_key] = before_user
        cost_tracker._memory[day_key] = before_day
        cost_tracker_mod._budget_holds.reset(token)


def _uploaded_session(*, file_size: int, user_id: int) -> UploadSession:
    return UploadSession(
        upload_id="u-size",
        token="t",
        chat_id=10,
        user_id=user_id,
        purpose="transcribe",
        media_kind="video",
        status="uploaded",
        created_at=0.0,
        expires_at=9999.0,
        object_key="uploads/big.bin",
        filename="clip.mp4",
        mime_type="video/mp4",
        file_size=file_size,
    )


@pytest.mark.asyncio
async def test_large_video_passes_budget_until_real_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """300 МБ не оцениваются как ~$8 аудио: резерв — пол, добор по длительности."""
    runtime._upload_process_sem = asyncio.Semaphore(2)
    user_id = 515151
    session = _uploaded_session(file_size=300 * 1024 * 1024, user_id=user_id)
    bot: Any = SimpleNamespace(
        send_message=AsyncMock(return_value=SimpleNamespace(delete=AsyncMock()))
    )
    store: Any = SimpleNamespace(update=AsyncMock())
    storage: Any = SimpleNamespace(
        config=SimpleNamespace(max_upload_mb=512),
        download_bytes=AsyncMock(return_value=b"video"),
        delete_object=AsyncMock(),
    )
    transcribe = AsyncMock(
        return_value=TranscriptionResult(
            text="",
            cost_usd=0.01,
            cost_rub=0.8,
            cost_source="cost_rub",
            provider_id="proxyapi",
            model="whisper-1",
        )
    )
    monkeypatch.setattr(rate_limiter, "check_media_limit", AsyncMock(return_value=True))
    monkeypatch.setattr(
        "app.media.file_magic.mime_compatible", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(
        runtime,
        "prepare_audio_for_stt",
        AsyncMock(return_value=[(b"mp3", "audio.mp3")]),
    )
    monkeypatch.setattr(
        "app.media.media_utils.audio_duration_seconds",
        AsyncMock(return_value=120.0),
    )
    monkeypatch.setattr(runtime.job_queue, "transcribe_audio", transcribe)

    monkeypatch.setenv("LLM_DAILY_BUDGET_USD", "1.5")
    monkeypatch.setenv("LLM_USER_DAILY_BUDGET_USD", "1.5")
    token = cost_tracker_mod._budget_holds.set(())
    try:
        await runtime._process_s3_upload_session(
            bot=bot,
            assistant=MagicMock(),
            topic_storage=MagicMock(),
            assistant_storage=MagicMock(),
            store=store,
            storage=storage,
            session=session,
        )
        assert session.status == "done"
        storage.download_bytes.assert_awaited()
        transcribe.assert_awaited()
        assert cost_tracker_mod._reserved_amount() == pytest.approx(0.0)
    finally:
        await cost_tracker.release_reservation(user_id)
        cost_tracker_mod._budget_holds.reset(token)


@pytest.mark.asyncio
async def test_long_audio_rejected_after_extract_before_stt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime._upload_process_sem = asyncio.Semaphore(2)
    user_id = 616161
    session = _uploaded_session(file_size=300 * 1024 * 1024, user_id=user_id)
    bot: Any = SimpleNamespace(send_message=AsyncMock())
    store: Any = SimpleNamespace(update=AsyncMock())
    storage: Any = SimpleNamespace(
        config=SimpleNamespace(max_upload_mb=512),
        download_bytes=AsyncMock(return_value=b"video"),
        delete_object=AsyncMock(),
    )
    transcribe = AsyncMock()
    monkeypatch.setattr(rate_limiter, "check_media_limit", AsyncMock(return_value=True))
    monkeypatch.setattr(
        "app.media.file_magic.mime_compatible", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(
        runtime,
        "prepare_audio_for_stt",
        AsyncMock(return_value=[(b"mp3", "audio.mp3")]),
    )
    monkeypatch.setattr(
        "app.media.media_utils.audio_duration_seconds",
        AsyncMock(return_value=10_000.0),
    )
    monkeypatch.setattr(runtime.job_queue, "transcribe_audio", transcribe)

    # Пол (~$0.0015) проходит, добор по 10 000 с — нет.
    monkeypatch.setenv("LLM_DAILY_BUDGET_USD", "0.02")
    monkeypatch.setenv("LLM_USER_DAILY_BUDGET_USD", "0.02")
    token = cost_tracker_mod._budget_holds.set(())
    try:
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
        storage.download_bytes.assert_awaited()
        transcribe.assert_not_called()
        assert cost_tracker_mod._reserved_amount() == pytest.approx(0.0)
    finally:
        await cost_tracker.release_reservation(user_id)
        cost_tracker_mod._budget_holds.reset(token)


@pytest.mark.asyncio
async def test_upload_status_deleted_when_processing_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime._upload_process_sem = asyncio.Semaphore(2)
    session = _uploaded_session(file_size=12_000, user_id=7)
    status = SimpleNamespace(delete=AsyncMock())
    bot: Any = SimpleNamespace(send_message=AsyncMock(return_value=status))
    store: Any = SimpleNamespace(update=AsyncMock())
    storage: Any = SimpleNamespace(
        config=SimpleNamespace(max_upload_mb=64),
        download_bytes=AsyncMock(side_effect=AssistAIError("s3 down")),
        delete_object=AsyncMock(),
    )
    monkeypatch.setattr(rate_limiter, "check_media_limit", AsyncMock(return_value=True))
    monkeypatch.setattr(cost_tracker, "check_budget", AsyncMock(return_value=None))
    monkeypatch.setattr(cost_tracker, "release_reservation", AsyncMock())

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
    status.delete.assert_awaited_once()
