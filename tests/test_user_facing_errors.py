"""User-facing errors must not leak provider/S3 payloads or URLs."""

from __future__ import annotations

from app.bot.constants import MSG_UPLOAD_PROCESS_FAILED
from app.llm.media_api import video_user_error


def test_video_user_errors_have_no_urls_or_payload_keys() -> None:
    for kind in ("failed", "poll", "download"):
        text = video_user_error(kind)
        assert "://" not in text
        assert "polling_url" not in text
        assert "unsigned_urls" not in text
        assert "{" not in text


def test_upload_process_user_message_is_generic() -> None:
    assert "://" not in MSG_UPLOAD_PROCESS_FAILED
    assert "S3" not in MSG_UPLOAD_PROCESS_FAILED
    assert "обработать" in MSG_UPLOAD_PROCESS_FAILED
