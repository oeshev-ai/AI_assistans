"""MIME signature sniffing for uploads."""

from __future__ import annotations

from app.media.file_magic import mime_compatible, sniff_mime


def test_sniff_mp3_id3() -> None:
    assert sniff_mime(b"ID3fake-mp3-bytes") == "audio/mpeg"


def test_sniff_wav() -> None:
    head = b"RIFF....WAVE" + b"\x00" * 20
    assert sniff_mime(head) == "audio/wav"


def test_rejects_pdf_disguised_as_mp3() -> None:
    assert not mime_compatible(
        filename="lecture.mp3",
        claimed_mime="audio/mpeg",
        head=b"%PDF-1.7 fake",
    )


def test_accepts_matching_mp3() -> None:
    assert mime_compatible(
        filename="lecture.mp3",
        claimed_mime="audio/mpeg",
        head=b"ID3fake-mp3-bytes",
    )
