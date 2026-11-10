"""Unit tests for job queue inline fallback."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import job_queue


@pytest.mark.asyncio
async def test_generate_image_inline_when_disabled() -> None:
    await job_queue.close_job_queue()
    job_queue._enabled = False
    job_queue._pool = None

    class FakeAssistant:
        async def generate_image(self, prompt, model=None, size="1024x1024"):
            return [f"img:{prompt}:{model}:{size}"]

    result = await job_queue.generate_image(
        FakeAssistant(), "cat", "model-a", "512x512"
    )
    assert result == ["img:cat:model-a:512x512"]


@pytest.mark.asyncio
async def test_transcribe_inline_when_disabled() -> None:
    await job_queue.close_job_queue()
    job_queue._enabled = False
    job_queue._pool = None

    class FakeAssistant:
        async def transcribe_audio(
            self, audio_bytes, filename="voice.ogg", model=None, language=None
        ):
            return f"heard:{audio_bytes.decode()}:{language}"

    text = await job_queue.transcribe_audio(FakeAssistant(), b"hi", language="ru")
    assert text == "heard:hi:ru"


@pytest.mark.asyncio
async def test_init_disabled_without_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARQ_ENABLED", raising=False)
    await job_queue.close_job_queue()
    await job_queue.init_job_queue("redis://localhost:6379/0")
    assert job_queue.is_queue_enabled() is False


@pytest.mark.asyncio
async def test_synthesize_speech_inline() -> None:
    await job_queue.close_job_queue()
    job_queue._enabled = False
    job_queue._pool = None

    speech = SimpleNamespace(audio=b"abc", filename="a.ogg", mime_type="audio/ogg")

    class FakeAssistant:
        async def synthesize_speech(
            self, text, model=None, voice=None, response_format=None
        ):
            return speech

    result = await job_queue.synthesize_speech(FakeAssistant(), "hello", voice="alloy")
    assert result.audio == b"abc"
