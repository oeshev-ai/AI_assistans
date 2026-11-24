"""Unit tests for job queue inline fallback."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import app.infra.job_queue as job_queue


@pytest.mark.asyncio
async def test_generate_image_inline_when_disabled() -> None:
    await job_queue.close_job_queue()
    job_queue._enabled = False
    job_queue._pool = None

    class FakeAssistant:
        async def generate_image(
            self, prompt, model=None, size="1024x1024", background=None
        ):
            return [f"img:{prompt}:{model}:{size}"]

    result = await job_queue.generate_image(
        FakeAssistant(), "cat", "model-a", "512x512"
    )
    assert result == ["img:cat:model-a:512x512"]


@pytest.mark.asyncio
async def test_generate_image_enqueue_none_fail_closed() -> None:
    from app.llm.errors import AssistAIError

    await job_queue.close_job_queue()
    job_queue._enabled = True

    class FakePool:
        async def enqueue_job(self, *args, **kwargs):
            return None

    job_queue._pool = FakePool()

    class FakeAssistant:
        async def generate_image(self, *args, **kwargs):
            raise AssertionError("must not fall back inline")

    with pytest.raises(AssistAIError, match="Очередь обработки"):
        await job_queue.generate_image(FakeAssistant(), "cat", "model-a", "512x512")
    job_queue._enabled = False
    job_queue._pool = None


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
    assert text.text == "heard:hi:ru"
    assert text.cost_usd is None


@pytest.mark.asyncio
async def test_init_disabled_without_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARQ_ENABLED", raising=False)
    await job_queue.close_job_queue()
    await job_queue.init_job_queue("redis://localhost:6379/0")
    assert job_queue.is_queue_enabled() is False


@pytest.mark.asyncio
async def test_init_rejects_short_arq_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARQ_ENABLED", "1")
    monkeypatch.setenv("POLZA_TOTAL_WAIT", "240")
    monkeypatch.setenv("ARQ_JOB_TIMEOUT", "180")
    await job_queue.close_job_queue()
    with pytest.raises(SystemExit):
        await job_queue.init_job_queue("redis://localhost:6379/0")


@pytest.mark.asyncio
async def test_init_fail_closed_without_redis_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARQ_ENABLED", "1")
    monkeypatch.setenv("POLZA_TOTAL_WAIT", "240")
    monkeypatch.setenv("ARQ_JOB_TIMEOUT", "450")
    await job_queue.close_job_queue()
    with pytest.raises(SystemExit):
        await job_queue.init_job_queue("")


@pytest.mark.asyncio
async def test_init_fail_closed_when_pool_create_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARQ_ENABLED", "1")
    monkeypatch.setenv("POLZA_TOTAL_WAIT", "240")
    monkeypatch.setenv("ARQ_JOB_TIMEOUT", "450")

    async def boom(_settings: object) -> None:
        raise ConnectionError("redis down")

    monkeypatch.setattr("arq.create_pool", boom)
    await job_queue.close_job_queue()
    with pytest.raises(SystemExit):
        await job_queue.init_job_queue("redis://localhost:6379/0")
    assert job_queue.is_queue_enabled() is False


def test_redis_settings_empty_password_is_none() -> None:
    settings = job_queue._redis_settings_from_url("redis://:@redis:6379/0")
    assert settings.password is None
    assert settings.host == "redis"


def test_image_result_timeout_covers_polza_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POLZA_TOTAL_WAIT", "240")
    monkeypatch.setenv("ARQ_JOB_TIMEOUT", "450")
    assert job_queue.image_result_timeout() == pytest.approx(450.0)
    assert job_queue.image_result_timeout(120.0) == pytest.approx(120.0)
    assert job_queue.audio_result_timeout() == pytest.approx(450.0)


def test_image_result_timeout_ignores_short_arq_job_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POLZA_TOTAL_WAIT", "240")
    monkeypatch.setenv("ARQ_JOB_TIMEOUT", "200")
    assert job_queue.image_result_timeout() == pytest.approx(450.0)


def test_validate_arq_image_timeout_rejects_short_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POLZA_TOTAL_WAIT", "240")
    monkeypatch.setenv("ARQ_JOB_TIMEOUT", "200")
    with pytest.raises(ValueError, match="ARQ_JOB_TIMEOUT"):
        job_queue.validate_arq_image_timeout()


def test_validate_arq_image_timeout_ok(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POLZA_TOTAL_WAIT", "240")
    monkeypatch.setenv("ARQ_JOB_TIMEOUT", "450")
    job_queue.validate_arq_image_timeout()


@pytest.mark.asyncio
async def test_synthesize_speech_inline() -> None:
    await job_queue.close_job_queue()
    job_queue._enabled = False
    job_queue._pool = None

    speech = SimpleNamespace(audio=b"abc", filename="a.ogg", mime_type="audio/ogg")

    class FakeAssistant:
        async def synthesize_speech(
            self, text, model=None, voice=None, response_format=None, role=None
        ):
            return speech

    result = await job_queue.synthesize_speech(FakeAssistant(), "hello", voice="alloy")
    assert result.audio == b"abc"


@pytest.mark.asyncio
async def test_synthesize_speech_enqueue_none_fail_closed() -> None:
    from app.llm.errors import AssistAIError

    await job_queue.close_job_queue()
    job_queue._enabled = True

    class FakePool:
        async def enqueue_job(self, *args, **kwargs):
            return None

    job_queue._pool = FakePool()

    class FakeAssistant:
        async def synthesize_speech(self, *args, **kwargs):
            raise AssertionError("must not fall back inline")

    with pytest.raises(AssistAIError, match="Очередь обработки"):
        await job_queue.synthesize_speech(FakeAssistant(), "hello")
    job_queue._enabled = False
    job_queue._pool = None


@pytest.mark.asyncio
async def test_transcribe_enqueue_none_fail_closed() -> None:
    from app.llm.errors import AssistAIError

    await job_queue.close_job_queue()
    job_queue._enabled = True

    class FakePool:
        async def set(self, *args, **kwargs):
            return True

        async def delete(self, *args, **kwargs):
            return 1

        async def enqueue_job(self, *args, **kwargs):
            return None

    job_queue._pool = FakePool()

    class FakeAssistant:
        async def transcribe_audio(self, *args, **kwargs):
            raise AssertionError("must not fall back inline")

    with pytest.raises(AssistAIError, match="Очередь обработки"):
        await job_queue.transcribe_audio(FakeAssistant(), b"hi")
    job_queue._enabled = False
    job_queue._pool = None


def test_payload_ttl_covers_job_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARQ_JOB_TIMEOUT", "900")
    monkeypatch.setenv("POLZA_TOTAL_WAIT", "240")
    assert job_queue._payload_ttl_seconds() == 1020


def test_payload_ttl_covers_queue_backlog(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARQ_JOB_TIMEOUT", "450")
    monkeypatch.setenv("ARQ_MAX_JOBS", "10")
    monkeypatch.setenv("POLZA_TOTAL_WAIT", "240")
    # 30 в очереди: волна in-progress + 3 пачки по 10 слотов.
    assert job_queue._payload_ttl_seconds(30) == 4 * 450 + 120
    assert job_queue._payload_ttl_seconds(0) == 600


@pytest.mark.asyncio
async def test_staged_payload_ttl_reads_queue_depth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARQ_JOB_TIMEOUT", "450")
    monkeypatch.setenv("ARQ_MAX_JOBS", "10")
    monkeypatch.setenv("POLZA_TOTAL_WAIT", "240")

    class Pool:
        async def zcard(self, key: str) -> int:
            assert key == "arq:queue"
            return 30

    assert await job_queue._staged_payload_ttl(Pool()) == 4 * 450 + 120


@pytest.mark.asyncio
async def test_staged_payload_ttl_ignores_depth_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARQ_JOB_TIMEOUT", "450")
    monkeypatch.setenv("ARQ_MAX_JOBS", "10")
    monkeypatch.setenv("POLZA_TOTAL_WAIT", "240")

    class Pool:
        async def zcard(self, _key: str) -> int:
            raise ConnectionError("redis down")

    assert await job_queue._staged_payload_ttl(Pool()) == 600
