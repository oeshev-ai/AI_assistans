"""User-facing errors must not leak provider/S3 payloads or URLs."""

from __future__ import annotations

from app.bot.constants import MSG_UPLOAD_PROCESS_FAILED
from app.bot.image_retry import IMAGE_EXPIRED_TEXT, IMAGE_KEEP_TEXT, IMAGE_RETRY_TEXT
from app.bot.video_retry import VIDEO_EXPIRED_TEXT, VIDEO_KEEP_TEXT, VIDEO_RETRY_TEXT
from app.llm.media_api import video_user_error


def test_video_user_errors_have_no_urls_or_payload_keys() -> None:
    for kind in ("failed", "poll", "download"):
        text = video_user_error(kind)
        assert "://" not in text
        assert "polling_url" not in text
        assert "unsigned_urls" not in text
        assert "{" not in text
        assert "не придёт" in text
        assert "новая генерация" in text


def test_image_retry_text_has_no_url() -> None:
    for text in (IMAGE_RETRY_TEXT, IMAGE_EXPIRED_TEXT):
        assert "://" not in text
        assert "polza" not in text.casefold()
        assert "{" not in text


def test_video_retry_text_promises_repeats_without_a_url() -> None:
    assert "повторяться" in VIDEO_RETRY_TEXT
    assert "новая генерация" in VIDEO_RETRY_TEXT.casefold()
    assert "не придёт" not in VIDEO_RETRY_TEXT
    assert "оплачен" in VIDEO_KEEP_TEXT
    assert "не запускается" in VIDEO_KEEP_TEXT
    assert "оплачена" in IMAGE_KEEP_TEXT
    for text in (
        VIDEO_RETRY_TEXT,
        VIDEO_EXPIRED_TEXT,
        VIDEO_KEEP_TEXT,
        IMAGE_KEEP_TEXT,
    ):
        assert "://" not in text
        assert "polza" not in text.casefold()
        assert "{" not in text


def test_upload_process_user_message_is_generic() -> None:
    assert "://" not in MSG_UPLOAD_PROCESS_FAILED
    assert "S3" not in MSG_UPLOAD_PROCESS_FAILED
    assert "обработать" in MSG_UPLOAD_PROCESS_FAILED
