"""Входы бота потребляют ведро один раз. Провайдер и додоставка — нет."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from respx import MockRouter

from app.infra.cost_tracker import set_billing_user
from app.infra.rate_limiter import rate_limiter
from app.llm.assist_ai import AssistAIClient
from app.llm.errors import AssistAIError
from app.llm.image_ops import ImageModelScope


@pytest.fixture
async def bucket_redis():
    fakeredis = pytest.importorskip("fakeredis.aioredis")
    client = fakeredis.FakeRedis(decode_responses=True)
    previous = rate_limiter._redis
    saved = dict(rate_limiter._image_buckets)
    rate_limiter.attach_redis(client)
    rate_limiter._image_buckets.clear()
    try:
        yield client
    finally:
        rate_limiter.attach_redis(previous)
        rate_limiter._image_buckets.clear()
        rate_limiter._image_buckets.update(saved)
        set_billing_user(None)
        await client.aclose()


async def _tokens(client: Any, user_id: int, operation: str) -> float | None:
    raw = await client.hget(f"rl:{operation}:{user_id}", "tokens")
    if raw is None:
        return None
    return float(raw)


@pytest.mark.asyncio
async def test_generate_tool_consumes_once_across_provider_retry(
    assist_client: AssistAIClient,
    bucket_redis: Any,
    respx_mock: MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("app.llm.image_ops._sleep", _no_sleep)
    monkeypatch.setattr("app.llm.transport._sleep", _no_sleep)
    route = respx_mock.post("https://api.proxyapi.ru/v1/images/generations").mock(
        side_effect=[
            httpx.Response(
                429,
                headers={"Retry-After": "0"},
                json={"error": {"code": "TOO_MANY_REQUESTS"}},
            ),
            httpx.Response(200, json={"data": [{"url": "https://cdn.example/ok.png"}]}),
            httpx.Response(
                429,
                headers={"Retry-After": "0"},
                json={"error": {"code": "TOO_MANY_REQUESTS"}},
            ),
            httpx.Response(
                200, json={"data": [{"url": "https://cdn.example/again.png"}]}
            ),
        ]
    )
    set_billing_user(42)
    tool = assist_client._tools_registry["generate_image"]
    with ImageModelScope(None):
        payload = await tool.execute(prompt="кот", size="1024x1024")
    assert "success" in payload
    assert await _tokens(bucket_redis, 42, "image_generate") == 14
    assert await _tokens(bucket_redis, 42, "image_edit") is None

    images = await assist_client.generate_image(
        "ещё кот",
        model="openai/gpt-image-2.6-flare",
        size="1024x1024",
    )
    assert images == ["https://cdn.example/again.png"]
    assert len(route.calls) == 4
    assert await _tokens(bucket_redis, 42, "image_generate") == 14

    with pytest.raises(AssistAIError):
        await assist_client.recover_paid_file("polza", "not-a-paid-job", kind="image")
    assert await _tokens(bucket_redis, 42, "image_generate") == 14
    assert await _tokens(bucket_redis, 42, "image_edit") is None


@pytest.mark.asyncio
async def test_edit_tool_consumes_edit_bucket_once(
    assist_client: AssistAIClient,
    bucket_redis: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.media.media_tools import (
        EditImageTool,
        MediaTurnScope,
        UserMedia,
        reset_user_media,
        set_user_media,
    )

    calls = 0

    async def fake_edit(*_args: object, **_kwargs: object) -> list[str]:
        nonlocal calls
        calls += 1
        return ["https://cdn.example/edited.png"]

    monkeypatch.setattr("app.infra.job_queue.edit_images", fake_edit)
    set_billing_user(42)
    tool = EditImageTool(assist_client)
    token = set_user_media(
        [
            UserMedia(
                kind="image",
                filename="a.png",
                mime_type="image/png",
                data=b"png",
            )
        ]
    )
    try:
        with MediaTurnScope():
            payload = await tool.execute(instruction="убери фон")
    finally:
        reset_user_media(token)
    assert "success" in payload
    assert calls == 1
    assert await _tokens(bucket_redis, 42, "image_edit") == 14
    assert await _tokens(bucket_redis, 42, "image_generate") is None


@pytest.mark.asyncio
async def test_photo_flow_generate_consumes_generate_bucket(
    bucket_redis: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.bot.photo_flow as flow
    import app.infra.job_queue as job_queue
    from app.bot.photo_router import PhotoDecision
    from app.storage.pending_media import (
        InMemoryPendingMediaStorage,
        PendingMedia,
        PendingMediaItem,
    )

    calls = {"generate": 0, "edit": 0}

    async def fake_generate(*_args: object, **_kwargs: object) -> list[str]:
        calls["generate"] += 1
        return ["https://cdn.example/new.png"]

    async def fake_edit(*_args: object, **_kwargs: object) -> list[str]:
        calls["edit"] += 1
        return ["https://cdn.example/ref.png"]

    async def send(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace()

    async def noop(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(job_queue, "generate_image", fake_generate)
    monkeypatch.setattr(job_queue, "edit_images", fake_edit)
    monkeypatch.setattr(flow.R, "_send_generated_image", send)
    monkeypatch.setattr(flow.R, "_reset_flow", noop)
    monkeypatch.setattr(flow, "_status", lambda *_a, **_k: _ready())
    monkeypatch.setattr(flow, "_drop_status", noop)
    monkeypatch.setattr(flow, "_remember_pair", noop)
    monkeypatch.setattr(flow.cost_tracker, "check_budget", noop)
    monkeypatch.setattr(flow.cost_tracker, "release_reservation", noop)
    monkeypatch.setenv("PHOTO_CONFIRM_MODE", "C")

    pending = InMemoryPendingMediaStorage()
    media = PendingMedia(
        chat_id=1,
        user_id=7,
        assistant_id="a",
        items=[PendingMediaItem(file_id="source", message_id=1)],
        caption="нарисуй постер",
    )

    class _Message:
        chat = SimpleNamespace(id=1)
        from_user = SimpleNamespace(id=7)

        async def answer(self, text: str, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(message_id=3)

    turn = SimpleNamespace(
        message=_Message(),
        assistant=SimpleNamespace(
            config=SimpleNamespace(
                default_image_model="openai/gpt-image-2.6-flare",
                default_image_edit_model="openai/gpt-image-2.6-flare",
                default_text_model="openai/gpt-4o",
            )
        ),
        topic_storage=None,
        assistant_storage=None,
        state=SimpleNamespace(),
        pending=pending,
    )
    profile = SimpleNamespace(
        image_model="",
        image_size="1024x1024",
        image_background="auto",
        text_model="",
        system_prompt="",
        image_as_file={},
    )
    await flow._generate(
        turn,
        media,
        profile,
        loaded=[],
        decision=PhotoDecision(intent="generate", instruction_en="poster"),
        topic_id="t",
    )
    assert calls == {"generate": 1, "edit": 0}
    assert await _tokens(bucket_redis, 7, "image_generate") == pytest.approx(
        14, abs=0.05
    )
    assert await _tokens(bucket_redis, 7, "image_edit") is None

    from app.media.media_tools import UserMedia

    loaded = [
        UserMedia(
            kind="image",
            filename="photo.jpg",
            mime_type="image/jpeg",
            data=b"jpeg",
        )
    ]
    await flow._generate(
        turn,
        media,
        profile,
        loaded=loaded,
        decision=PhotoDecision(
            intent="generate",
            instruction_en="poster from photo",
            instruction_ru="постер",
        ),
        topic_id="t",
    )
    assert calls == {"generate": 1, "edit": 1}
    assert await _tokens(bucket_redis, 7, "image_generate") == pytest.approx(
        13, abs=0.05
    )
    assert await _tokens(bucket_redis, 7, "image_edit") is None


async def _ready() -> SimpleNamespace:
    return SimpleNamespace(message_id=1)
