"""TTS response_format normalization."""

from __future__ import annotations

from app.llm.media_api import _speech_file, normalize_tts_format


def test_normalize_known_formats() -> None:
    assert normalize_tts_format("MP3") == "mp3"
    assert normalize_tts_format("opus") == "opus"
    assert normalize_tts_format("wav") == "wav"


def test_normalize_unknown_falls_back() -> None:
    assert normalize_tts_format("weird") == "opus"
    assert normalize_tts_format("") == "opus"
    assert normalize_tts_format(None) == "opus"


def test_speech_file_labels_match_format() -> None:
    assert _speech_file("mp3") == ("answer.mp3", "audio/mpeg")
    assert _speech_file("wav") == ("answer.wav", "audio/wav")
    assert _speech_file("nope") == ("answer.ogg", "audio/ogg")
