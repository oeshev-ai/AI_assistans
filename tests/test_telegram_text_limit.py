"""Лимит Telegram считается в UTF-16 code units, не в code points."""

from __future__ import annotations

from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import SendMessage

from app.bot import runtime
from app.bot.models import _fit_screen
from app.bot.telegram_text import (
    TELEGRAM_TEXT_LIMIT,
    TELEGRAM_TEXT_SAFE,
    clip_telegram_text,
    split_telegram_text,
    telegram_text_units,
)

EMOJI = "😀"


def _too_long(text: str) -> TelegramBadRequest:
    return TelegramBadRequest(
        method=SendMessage(chat_id=1, text=text),
        message="Bad Request: message is too long",
    )


def test_emoji_is_two_utf16_units() -> None:
    assert len(EMOJI) == 1
    assert telegram_text_units(EMOJI) == len(EMOJI.encode("utf-16-le")) // 2
    assert telegram_text_units(EMOJI) == 2


def test_code_point_slice_of_emoji_exceeds_api_limit() -> None:
    text = EMOJI * TELEGRAM_TEXT_LIMIT
    assert len(text) == TELEGRAM_TEXT_LIMIT
    assert telegram_text_units(text) == TELEGRAM_TEXT_LIMIT * 2
    clipped = clip_telegram_text(text)
    assert telegram_text_units(clipped) <= TELEGRAM_TEXT_SAFE
    assert text.startswith(clipped)


def test_clip_does_not_split_surrogate_pair() -> None:
    assert clip_telegram_text(f"{EMOJI}a", 1) == ""
    assert clip_telegram_text(f"{EMOJI}a", 2) == EMOJI
    assert clip_telegram_text(f"a{EMOJI}", 2) == "a"
    assert clip_telegram_text(EMOJI * 3, 4) == EMOJI * 2
    assert clip_telegram_text("abc", 0) == ""
    assert clip_telegram_text("", 10) == ""


def test_split_covers_text_without_overlap() -> None:
    text = f"a{EMOJI}b" * 100
    parts = split_telegram_text(text, 10)
    assert "".join(parts) == text
    assert parts
    assert all(telegram_text_units(part) <= 10 for part in parts)


def test_fit_screen_counts_emoji_units() -> None:
    fitted = _fit_screen(EMOJI * TELEGRAM_TEXT_SAFE)
    assert fitted.endswith("…")
    assert telegram_text_units(fitted) <= TELEGRAM_TEXT_SAFE


async def test_send_text_splits_emoji_under_code_point_limit() -> None:
    # 2500 эмодзи — это 2500 code points (старый лимит 4096 их пропускал)
    # и 5000 UTF-16 units (Bot API такой текст отвергает).
    text = EMOJI * 2500
    assert len(text) < TELEGRAM_TEXT_LIMIT
    assert telegram_text_units(text) > TELEGRAM_TEXT_LIMIT
    sent: list[str] = []

    class _Message:
        async def answer(self, value: str) -> None:
            sent.append(value)

    await runtime._send_text(_Message(), text)  # type: ignore[arg-type]
    assert "".join(sent) == text
    assert len(sent) >= 2
    assert all(telegram_text_units(part) <= TELEGRAM_TEXT_SAFE for part in sent)


async def test_edit_stream_and_overflow_share_utf16_cut() -> None:
    text = EMOJI * 2500
    edited: list[str] = []
    sent: list[str] = []

    class _Status:
        async def edit_text(self, value: str) -> None:
            edited.append(value)

    class _Message:
        async def answer(self, value: str) -> None:
            sent.append(value)

    shown = await runtime._edit_stream(_Status(), text, "")  # type: ignore[arg-type]
    head = clip_telegram_text(text, TELEGRAM_TEXT_SAFE)
    assert edited == [head]
    assert shown == head
    assert telegram_text_units(shown) <= TELEGRAM_TEXT_SAFE
    await runtime._send_overflow(_Message(), text)  # type: ignore[arg-type]
    assert shown + "".join(sent) == text


async def test_send_text_retries_when_telegram_says_too_long(monkeypatch) -> None:
    monkeypatch.setattr(runtime, "TELEGRAM_TEXT_SAFE", 8)
    sent: list[str] = []
    failed = False

    class _Message:
        async def answer(self, value: str) -> None:
            nonlocal failed
            sent.append(value)
            if not failed:
                failed = True
                raise _too_long(value)

    await runtime._send_text(_Message(), "abcdefghij")  # type: ignore[arg-type]
    assert sent[0] == "abcdefgh"
    assert "".join(sent[1:]) == "abcdefghij"


async def test_send_text_swallows_other_bad_request(monkeypatch) -> None:
    monkeypatch.setattr(runtime, "TELEGRAM_TEXT_SAFE", 4)
    sent: list[str] = []

    class _Message:
        async def answer(self, value: str) -> None:
            sent.append(value)
            raise TelegramBadRequest(
                method=SendMessage(chat_id=1, text=value),
                message="Bad Request: chat not found",
            )

    await runtime._send_text(_Message(), "abcdef")  # type: ignore[arg-type]
    assert sent == ["abcd"]


async def test_send_text_blank_becomes_placeholder() -> None:
    sent: list[str] = []

    class _Message:
        async def answer(self, value: str) -> None:
            sent.append(value)

    await runtime._send_text(_Message(), "   ")  # type: ignore[arg-type]
    assert sent == ["Пустой ответ."]
