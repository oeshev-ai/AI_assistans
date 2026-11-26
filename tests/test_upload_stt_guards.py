"""Guards for large-upload STT: budget estimate and concurrency helpers."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiogram.exceptions import TelegramNetworkError
from aiogram.methods import SendMessage

import app.bot.runtime as runtime
from app.bot.constants import MSG_UPLOAD_PROCESS_FAILED
from app.bot.telegram_text import TELEGRAM_TEXT_SAFE, split_telegram_text
from app.infra import cost_tracker as cost_tracker_mod
from app.infra.cost_tracker import BudgetExceededError, cost_tracker
from app.infra.rate_limiter import rate_limiter
from app.llm.errors import AssistAIError
from app.llm.media_api import TranscriptionResult
from app.media.media_utils import MediaProcessingError
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
        download_file=AsyncMock(),
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
    storage.download_file.assert_not_called()
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
        download_file=AsyncMock(),
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
    storage.download_file.assert_not_called()
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
        send_message=AsyncMock(
            return_value=SimpleNamespace(delete=AsyncMock(), edit_text=AsyncMock())
        ),
        send_chat_action=AsyncMock(),
    )
    store: Any = SimpleNamespace(update=AsyncMock())
    storage: Any = SimpleNamespace(
        config=SimpleNamespace(max_upload_mb=64),
        download_file=AsyncMock(return_value=b"audio"),
        delete_object=AsyncMock(),
    )
    monkeypatch.setattr(rate_limiter, "check_media_limit", AsyncMock(return_value=True))
    monkeypatch.setattr(
        "app.media.file_magic.mime_compatible", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(
        runtime,
        "prepare_audio_file_for_stt",
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
async def test_verified_mime_skips_second_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime._upload_process_sem = asyncio.Semaphore(2)
    user_id = 717171
    session = _uploaded_session(file_size=1200, user_id=user_id)
    session.verified_mime = session.mime_type
    probe = AsyncMock(return_value=True)
    bot: Any = SimpleNamespace(
        send_message=AsyncMock(
            return_value=SimpleNamespace(delete=AsyncMock(), edit_text=AsyncMock())
        ),
        send_chat_action=AsyncMock(),
    )
    store: Any = SimpleNamespace(update=AsyncMock())
    storage: Any = SimpleNamespace(
        config=SimpleNamespace(max_upload_mb=512),
        download_file=AsyncMock(),
        delete_object=AsyncMock(),
    )
    monkeypatch.setattr(rate_limiter, "check_media_limit", AsyncMock(return_value=True))
    monkeypatch.setattr(cost_tracker, "check_budget", AsyncMock())
    monkeypatch.setattr(cost_tracker, "release_reservation", AsyncMock())
    monkeypatch.setattr(cost_tracker, "record", AsyncMock())
    monkeypatch.setattr("app.media.file_magic.mime_compatible", probe)
    monkeypatch.setattr(
        runtime,
        "prepare_audio_file_for_stt",
        AsyncMock(return_value=[(b"mp3", "audio.mp3")]),
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
                cost_usd=0.01,
                cost_rub=0.8,
                cost_source="cost_rub",
                provider_id="proxyapi",
                model="whisper-1",
            )
        ),
    )
    monkeypatch.setattr(runtime, "_delete_chat_messages", AsyncMock())

    await runtime._process_s3_upload_session(
        bot=bot,
        assistant=MagicMock(),
        topic_storage=MagicMock(append_messages=AsyncMock(), max_length=10),
        assistant_storage=MagicMock(),
        store=store,
        storage=storage,
        session=session,
    )
    probe.assert_not_called()
    runtime.prepare_audio_file_for_stt.assert_awaited()


@pytest.mark.asyncio
async def test_large_video_passes_budget_until_real_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """300 МБ не оцениваются как ~$8 аудио: резерв — пол, добор по длительности."""
    runtime._upload_process_sem = asyncio.Semaphore(2)
    user_id = 515151
    session = _uploaded_session(file_size=300 * 1024 * 1024, user_id=user_id)
    bot: Any = SimpleNamespace(
        send_message=AsyncMock(
            return_value=SimpleNamespace(delete=AsyncMock(), edit_text=AsyncMock())
        ),
        send_chat_action=AsyncMock(),
    )
    store: Any = SimpleNamespace(update=AsyncMock())
    storage: Any = SimpleNamespace(
        config=SimpleNamespace(max_upload_mb=512),
        download_file=AsyncMock(return_value=b"video"),
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
        "prepare_audio_file_for_stt",
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
        storage.download_file.assert_awaited()
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
    bot: Any = SimpleNamespace(
        send_message=AsyncMock(),
        send_chat_action=AsyncMock(),
    )
    store: Any = SimpleNamespace(update=AsyncMock())
    storage: Any = SimpleNamespace(
        config=SimpleNamespace(max_upload_mb=512),
        download_file=AsyncMock(return_value=b"video"),
        delete_object=AsyncMock(),
    )
    transcribe = AsyncMock()
    monkeypatch.setattr(rate_limiter, "check_media_limit", AsyncMock(return_value=True))
    monkeypatch.setattr(
        "app.media.file_magic.mime_compatible", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(
        runtime,
        "prepare_audio_file_for_stt",
        AsyncMock(return_value=[(b"mp3", "audio.mp3")]),
    )
    monkeypatch.setattr(
        "app.media.media_utils.audio_duration_seconds",
        AsyncMock(return_value=10_000.0),
    )
    monkeypatch.setattr(runtime.job_queue, "transcribe_audio", transcribe)

    # Пол (~$0.0015) проходит, добор по 10 000 с — нет.
    # Расход других тестов в общем счётчике не должен съедать этот потолок.
    monkeypatch.setenv("LLM_DAILY_BUDGET_USD", "0.02")
    monkeypatch.setenv("LLM_USER_DAILY_BUDGET_USD", "0.02")
    day = cost_tracker._day_key()
    day_key = f"cost:day:{day}"
    user_key = f"cost:user:{user_id}:day:{day}"
    before_day = cost_tracker._memory.get(day_key, 0.0)
    before_user = cost_tracker._memory.get(user_key, 0.0)
    cost_tracker._memory[day_key] = 0.0
    cost_tracker._memory[user_key] = 0.0
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
        storage.download_file.assert_awaited()
        transcribe.assert_not_called()
        assert cost_tracker_mod._reserved_amount() == pytest.approx(0.0)
    finally:
        await cost_tracker.release_reservation(user_id)
        cost_tracker._memory[day_key] = before_day
        cost_tracker._memory[user_key] = before_user
        cost_tracker_mod._budget_holds.reset(token)


@pytest.mark.asyncio
async def test_upload_status_deleted_when_processing_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime._upload_process_sem = asyncio.Semaphore(2)
    session = _uploaded_session(file_size=12_000, user_id=7)
    status = SimpleNamespace(delete=AsyncMock(), edit_text=AsyncMock())
    bot: Any = SimpleNamespace(
        send_message=AsyncMock(return_value=status),
        send_chat_action=AsyncMock(),
    )
    store: Any = SimpleNamespace(update=AsyncMock())
    storage: Any = SimpleNamespace(
        config=SimpleNamespace(max_upload_mb=64),
        download_file=AsyncMock(side_effect=AssistAIError("s3 down")),
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


@pytest.mark.asyncio
async def test_failed_prepare_deletes_the_downloaded_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime._upload_process_sem = asyncio.Semaphore(2)
    session = _uploaded_session(file_size=12_000, user_id=7)
    session.verified_mime = session.mime_type
    status = SimpleNamespace(delete=AsyncMock(), edit_text=AsyncMock())
    downloaded: list[Path] = []

    async def _download(_key: str, dest: Path, **_kwargs: object) -> None:
        dest.write_bytes(b"not-a-container")
        downloaded.append(dest)

    async def _prepare(path: Path, **_kwargs: object) -> None:
        assert path.is_file()
        raise MediaProcessingError("битый контейнер")

    bot: Any = SimpleNamespace(
        send_message=AsyncMock(return_value=status),
        send_chat_action=AsyncMock(),
    )
    store: Any = SimpleNamespace(update=AsyncMock())
    storage: Any = SimpleNamespace(
        config=SimpleNamespace(max_upload_mb=64),
        download_file=AsyncMock(side_effect=_download),
        delete_object=AsyncMock(),
    )
    monkeypatch.setattr(rate_limiter, "check_media_limit", AsyncMock(return_value=True))
    monkeypatch.setattr(cost_tracker, "check_budget", AsyncMock(return_value=None))
    monkeypatch.setattr(cost_tracker, "release_reservation", AsyncMock())
    monkeypatch.setattr(runtime, "prepare_audio_file_for_stt", _prepare)

    await runtime._process_s3_upload_session(
        bot=bot,
        assistant=MagicMock(),
        topic_storage=MagicMock(),
        assistant_storage=MagicMock(),
        store=store,
        storage=storage,
        session=session,
    )

    assert downloaded
    assert not downloaded[0].exists()
    assert session.status == "failed"
    assert session.error == "битый контейнер"


@pytest.mark.asyncio
async def test_upload_progress_names_each_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime._upload_process_sem = asyncio.Semaphore(2)
    session = _uploaded_session(file_size=1200, user_id=818181)
    session.media_kind = "audio"
    session.filename = "talk.mp3"
    session.mime_type = "audio/mpeg"
    session.verified_mime = session.mime_type
    status = SimpleNamespace(delete=AsyncMock(), edit_text=AsyncMock())
    bot: Any = SimpleNamespace(
        send_message=AsyncMock(return_value=status),
        send_chat_action=AsyncMock(),
    )

    async def _download(*_args: object, **_kwargs: object) -> None:
        await asyncio.sleep(0)

    store: Any = SimpleNamespace(update=AsyncMock())
    storage: Any = SimpleNamespace(
        config=SimpleNamespace(max_upload_mb=512),
        download_file=AsyncMock(side_effect=_download),
        delete_object=AsyncMock(),
    )
    monkeypatch.setattr(rate_limiter, "check_media_limit", AsyncMock(return_value=True))
    monkeypatch.setattr(cost_tracker, "check_budget", AsyncMock())
    monkeypatch.setattr(cost_tracker, "release_reservation", AsyncMock())
    monkeypatch.setattr(cost_tracker, "record", AsyncMock())
    monkeypatch.setattr(
        runtime,
        "prepare_audio_file_for_stt",
        AsyncMock(return_value=[(b"mp3", f"part{i}.mp3") for i in range(5)]),
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
                cost_usd=0.01,
                cost_rub=0.8,
                cost_source="cost_rub",
                provider_id="proxyapi",
                model="whisper-1",
            )
        ),
    )
    monkeypatch.setattr(runtime, "_delete_chat_messages", AsyncMock())

    await runtime._process_s3_upload_session(
        bot=bot,
        assistant=MagicMock(),
        topic_storage=MagicMock(),
        assistant_storage=MagicMock(),
        store=store,
        storage=storage,
        session=session,
    )

    opened = bot.send_message.await_args_list[0].args[1]
    assert "Скачиваю" in opened
    edited = [call.args[0] for call in status.edit_text.await_args_list]
    assert any("Готовлю аудио" in text for text in edited)
    assert "Файл «talk.mp3» получен. Часть 2 из 5…" in edited
    assert "Файл «talk.mp3» получен. Часть 5 из 5…" in edited
    bot.send_chat_action.assert_awaited()


def _telegram_down(chat_id: int = 10) -> TelegramNetworkError:
    return TelegramNetworkError(
        method=SendMessage(chat_id=chat_id, text="x"),
        message="Server disconnected",
    )


def _patch_paid_transcript(
    monkeypatch: pytest.MonkeyPatch,
    *,
    text: str,
    keep: bool,
) -> tuple[UploadSession, Any, SimpleNamespace]:
    """Сессия доходит до отправки transcript.txt. keep — ответ бакета доставки."""
    runtime._upload_process_sem = asyncio.Semaphore(2)
    session = _uploaded_session(file_size=1200, user_id=919191)
    session.media_kind = "audio"
    session.filename = "talk.mp3"
    session.mime_type = "audio/mpeg"
    session.verified_mime = session.mime_type
    topic = SimpleNamespace(
        ensure_active=AsyncMock(return_value="topic"),
        append_messages=AsyncMock(),
        max_length=20,
    )
    status = SimpleNamespace(delete=AsyncMock(), edit_text=AsyncMock())
    bot: Any = SimpleNamespace(
        send_message=AsyncMock(return_value=status),
        send_document=AsyncMock(side_effect=_telegram_down()),
        send_chat_action=AsyncMock(),
        delete_message=AsyncMock(),
    )
    store: Any = SimpleNamespace(update=AsyncMock())
    storage: Any = SimpleNamespace(
        config=SimpleNamespace(max_upload_mb=512),
        download_file=AsyncMock(),
        delete_object=AsyncMock(),
        upload_bytes=AsyncMock(),
    )
    monkeypatch.setattr(rate_limiter, "check_media_limit", AsyncMock(return_value=True))
    monkeypatch.setattr(cost_tracker, "check_budget", AsyncMock())
    monkeypatch.setattr(cost_tracker, "release_reservation", AsyncMock())
    monkeypatch.setattr(cost_tracker, "record", AsyncMock())
    monkeypatch.setattr(
        runtime,
        "prepare_audio_file_for_stt",
        AsyncMock(return_value=[(b"mp3", "talk.mp3")]),
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
                text=text,
                cost_usd=0.01,
                cost_rub=0.8,
                cost_source="cost_rub",
                provider_id="proxyapi",
                model="whisper-1",
            )
        ),
    )
    monkeypatch.setattr(runtime, "_delete_chat_messages", AsyncMock())
    monkeypatch.setattr(
        "app.bot.file_retry.keep_paid_bytes", AsyncMock(return_value=keep)
    )
    assistant_storage = SimpleNamespace(
        get_active_assistant=AsyncMock(return_value=SimpleNamespace(id="asst"))
    )
    return (
        session,
        bot,
        SimpleNamespace(
            topic=topic,
            store=store,
            storage=storage,
            assistant_storage=assistant_storage,
            status=status,
        ),
    )


@pytest.mark.asyncio
async def test_unkept_transcript_reaches_chat_and_session_stays_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = "оплаченный текст"
    session, bot, parts = _patch_paid_transcript(monkeypatch, text=text, keep=False)

    await runtime._process_s3_upload_session(
        bot=bot,
        assistant=MagicMock(),
        topic_storage=parts.topic,
        assistant_storage=parts.assistant_storage,
        store=parts.store,
        storage=parts.storage,
        session=session,
    )

    sent = [call.args[1] for call in bot.send_message.await_args_list]
    assert text in sent
    assert runtime._PAID_TRANSCRIPT_INTRO in sent
    assert MSG_UPLOAD_PROCESS_FAILED not in sent
    assert session.status == "done"
    assert session.error is None
    stored = parts.topic.append_messages.await_args.args[3][1]["content"]
    assert "отправлена текстом" in stored
    assert text not in stored


@pytest.mark.asyncio
async def test_rejected_transcript_history_is_an_excerpt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = "а" * 20_000
    session, bot, parts = _patch_paid_transcript(monkeypatch, text=text, keep=False)
    opened = {"done": False}

    async def _send_message(_chat_id: int, _payload: str) -> object:
        if not opened["done"]:
            opened["done"] = True
            return parts.status
        raise _telegram_down()

    bot.send_message = AsyncMock(side_effect=_send_message)

    await runtime._process_s3_upload_session(
        bot=bot,
        assistant=MagicMock(),
        topic_storage=parts.topic,
        assistant_storage=parts.assistant_storage,
        store=parts.store,
        storage=parts.storage,
        session=session,
    )

    assert session.status == "done"
    assert session.error is None
    stored = parts.topic.append_messages.await_args.args[3][1]["content"]
    assert text not in stored
    assert text[:80] in stored
    assert len(stored) < runtime._TRANSCRIPT_HISTORY_CHARS + 400
    assert "transcript.txt" in stored
    parked = parts.storage.upload_bytes.await_args.kwargs
    assert parked["data"] == text.encode()
    assert parked["object_key"].endswith("/transcript.txt")
    assert parked["object_key"] != session.object_key
    assert MSG_UPLOAD_PROCESS_FAILED not in {
        call.args[1] for call in bot.send_message.await_args_list if len(call.args) > 1
    }


@pytest.mark.asyncio
async def test_unkept_transcript_chat_stops_after_five_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = "".join(
        f"#{index:02d}#" + ("x" * (TELEGRAM_TEXT_SAFE - 4)) for index in range(6)
    )
    session, bot, parts = _patch_paid_transcript(monkeypatch, text=text, keep=False)

    await runtime._process_s3_upload_session(
        bot=bot,
        assistant=MagicMock(),
        topic_storage=parts.topic,
        assistant_storage=parts.assistant_storage,
        store=parts.store,
        storage=parts.storage,
        session=session,
    )

    chunks = split_telegram_text(text, TELEGRAM_TEXT_SAFE)
    sent = [call.args[1] for call in bot.send_message.await_args_list]
    assert chunks[0] in sent
    assert chunks[runtime._TRANSCRIPT_CHAT_CHUNKS - 1] in sent
    assert chunks[runtime._TRANSCRIPT_CHAT_CHUNKS] not in sent
    assert runtime._PAID_TRANSCRIPT_MORE in sent
    stored = parts.topic.append_messages.await_args.args[3][1]["content"]
    assert text not in stored
    assert len(stored) < runtime._TRANSCRIPT_HISTORY_CHARS + 400
    assert parts.storage.upload_bytes.await_args.kwargs["data"] == text.encode()
    assert session.status == "done"
