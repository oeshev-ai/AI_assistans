"""Оценка входа, лимиты моделей, история персоны и отказ слишком большого хода."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, create_autospec

import httpx
import pytest
from aiogram.types import Message
from pydantic import ValidationError
from respx import MockRouter

from app.bot import models as M
from app.bot import runtime as R
from app.bot.handlers.settings import on_set_history
from app.infra.cost_tracker import chat_cost_usd, cost_tracker, prefer_cost_rub
from app.infra.rate_limiter import RateLimiter, effective_limits
from app.llm.assist_ai import AssistAIClient
from app.llm.chat_runtime import _apply_generation_options, _needs_input_retry
from app.llm.context import content_chars, estimate_tokens
from app.llm.errors import AssistAIError, fallback_reason
from app.llm.models_config import (
    AVAILABLE_TEXT_MODELS,
    MODEL_LIMITS,
    ModelLimits,
    validate_model_limits,
)
from app.storage.assistant_storage import AssistantProfile, InMemoryAssistantStorage
from app.storage.message_trim import shrink_to_token_limit, trim_messages
from app.storage.topic_history import InMemoryTopicStorage

_CHAT_URL = "https://api.proxyapi.ru/v1/chat/completions"


def _profile(**extra: Any) -> AssistantProfile:
    payload: dict[str, Any] = {
        "id": "persona",
        "name": "Стандартный",
        "system_prompt": "Ты тестовый помощник.",
        "voice_mode": "off",
    }
    payload.update(extra)
    return AssistantProfile(**payload)


class _Chat:
    id = 7


class _User:
    id = 11


class _Status:
    def __init__(self, edits: list[str]) -> None:
        self.edits = edits
        self.message_id = 42
        self.deleted = False

    async def edit_text(self, text: str) -> None:
        self.edits.append(text)

    async def delete(self) -> None:
        self.deleted = True


class _Message:
    def __init__(self) -> None:
        self.chat = _Chat()
        self.from_user = _User()
        self.bot = None
        self.edits: list[str] = []
        self.replies: list[str] = []
        self.status = _Status(self.edits)

    async def answer(self, text: str, reply_markup: object = None) -> _Status:
        del reply_markup
        self.replies.append(text)
        return self.status


class _Profiles:
    def __init__(self, profile: AssistantProfile) -> None:
        self.profile = profile

    async def get_active_assistant(self, chat_id: int) -> AssistantProfile:
        del chat_id
        return self.profile


class _State:
    def __init__(self) -> None:
        self.data: dict[str, object] = {}

    async def get_data(self) -> dict[str, object]:
        return self.data

    async def update_data(self, payload: dict[str, object]) -> dict[str, object]:
        self.data.update(payload)
        return self.data


def _patch_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _budget(*_args: object, **_kwargs: object) -> None:
        return None

    async def _release(*_args: object, **_kwargs: object) -> float:
        return 0.0

    monkeypatch.setattr(R.cost_tracker, "check_budget", _budget)
    monkeypatch.setattr(R.cost_tracker, "release_reservation", _release)
    monkeypatch.setattr(cost_tracker, "check_budget", _budget)
    monkeypatch.setattr(cost_tracker, "record", _budget)


def _image(payload: str) -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{payload}"},
    }


def test_multimodal_estimate_ignores_base64_length() -> None:
    huge = "A" * 50_000
    tiny = "AA"
    left = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "look"}, _image(huge)],
        }
    ]
    right = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "look"}, _image(tiny)],
        }
    ]
    assert content_chars(left[0]["content"]) == content_chars(right[0]["content"])
    assert estimate_tokens(left, chars_per_token=2.5) == estimate_tokens(
        right, chars_per_token=2.5
    )
    assert huge not in str(content_chars(left[0]["content"]))


@pytest.mark.asyncio
async def test_vision_budget_reserve_matches_estimate_without_base64(
    assist_client: AssistAIClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, float] = {}

    async def check_budget(*_args: object, **kwargs: object) -> None:
        estimated = kwargs.get("estimated")
        assert isinstance(estimated, float)
        seen["estimated"] = estimated

    monkeypatch.setattr(cost_tracker, "check_budget", check_budget)
    content = [
        {"type": "text", "text": "look"},
        _image("A" * 20_000),
    ]
    messages = [{"role": "user", "content": content}]
    await assist_client._ensure_round_budget(messages)
    expected = cost_tracker.estimate_chat_cost(
        input_chars=max(content_chars(content), 200),
        output_chars=400,
    )
    assert seen["estimated"] == expected
    with_rub, source, rub = chat_cost_usd(
        usage={"cost_rub": 3.04},
        provider=None,
        rub_per_usd=304.0,
        input_chars=10**9,
        output_chars=10,
    )
    same, _, _ = chat_cost_usd(
        usage={"cost_rub": 3.04},
        provider=None,
        rub_per_usd=304.0,
        input_chars=1,
        output_chars=1,
    )
    assert with_rub == same
    assert source == "cost_rub"
    assert rub == 3.04
    settled, settled_source, _ = prefer_cost_rub([(with_rub, source, rub)], 99.0)
    assert settled_source == "cost_rub"
    assert settled == with_rub


def test_model_limits_cover_catalog_and_clamp() -> None:
    assert set(MODEL_LIMITS) == set(AVAILABLE_TEXT_MODELS)
    assert len(MODEL_LIMITS) == 8
    assert MODEL_LIMITS["google/gemini-4-argon"].max_output == 66_000
    broken = dict(MODEL_LIMITS)
    broken.pop("google/gemini-4-argon")
    with pytest.raises(RuntimeError, match="gemini-4-argon"):
        validate_model_limits(broken)
    payload: dict[str, Any] = {"model": "google/gemini-4-argon"}
    _apply_generation_options(
        payload,
        response_format=None,
        temperature=None,
        max_tokens=200_000,
        catalog_model="google/gemini-4-argon",
    )
    assert payload["max_tokens"] == 66_000
    assert ModelLimits(128_000, 8_192).max_output == 8_192


def test_profile_without_history_field_loads_as_default() -> None:
    raw = {
        "id": "old",
        "name": "Старый",
        "system_prompt": "Ты тестовый помощник.",
    }
    profile = AssistantProfile.model_validate(raw)
    assert profile.history_length is None
    assert M._history_limit(profile) == 20


def test_history_length_rejects_out_of_range() -> None:
    profile = _profile()
    with pytest.raises(ValidationError):
        profile.validated_update({"history_length": 3})
    with pytest.raises(ValidationError):
        profile.validated_update({"history_length": 201})


def test_trim_keeps_system_user_tool_pair_and_protect_tail() -> None:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "old " * 50},
        {"role": "assistant", "content": "gone"},
        {"role": "user", "content": "keep"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "ping", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ]
    out = trim_messages(messages, 4, max_chars=80, protect_tail=3)
    assert out[0]["role"] == "system"
    assert out[-3]["content"] == "keep"
    assert out[-2]["role"] == "assistant"
    assert out[-1]["role"] == "tool"
    assert "old" not in json.dumps(out, ensure_ascii=False)


def test_shrink_drops_history_not_current_round() -> None:
    old = "O" * 4000
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": old},
        {"role": "assistant", "content": old},
        {"role": "user", "content": "сейчас"},
        {"role": "assistant", "content": ""},
        {"role": "tool", "tool_call_id": "c1", "content": "T" * 100},
    ]
    out = shrink_to_token_limit(
        messages,
        token_limit=1000,
        chars_per_token=2.5,
        protect_tail=3,
    )
    blob = json.dumps(out, ensure_ascii=False)
    assert "сейчас" in blob
    assert "T" * 100 in blob
    assert old not in blob


@pytest.mark.asyncio
async def test_two_personas_keep_separate_history_lengths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_budget(monkeypatch)
    storage = InMemoryTopicStorage(max_length=80)
    long = _profile(id="long", history_length=40)
    short = _profile(id="short", history_length=20)

    async def fill(profile: AssistantProfile, text: str) -> None:
        topic = await storage.ensure_active(1, profile.id, "Общая")
        for index in range(30):
            await storage.append_message(
                1,
                profile.id,
                topic,
                {"role": "user", "content": f"{text}-{index}"},
                M._history_limit(profile),
            )

    await fill(long, "L")
    await fill(short, "S")
    long_topic = await storage.get_active_topic(1, long.id)
    short_topic = await storage.get_active_topic(1, short.id)
    assert long_topic is not None and short_topic is not None
    assert len(await storage.get_history(1, long.id, long_topic)) == 30
    assert len(await storage.get_history(1, short.id, short_topic)) == 20


@pytest.mark.asyncio
async def test_lower_limit_cuts_on_read_and_raise_does_not_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_budget(monkeypatch)
    storage = InMemoryTopicStorage(max_length=80)
    profile = _profile(history_length=40)
    topic = await storage.ensure_active(7, profile.id, "Общая")
    for index in range(30):
        await storage.append_message(
            7,
            profile.id,
            topic,
            {"role": "user", "content": f"old-{index}"},
            40,
        )
    tight = profile.validated_update({"history_length": 10})
    prepared = await R._prepare_messages(7, tight.id, "новое", topic, storage, tight)
    dialog = [item for item in prepared if item["role"] != "system"]
    assert len(dialog) <= 10
    assert dialog[-1]["content"] == "новое"
    stored = await storage.get_history(7, tight.id, topic)
    assert len(stored) <= 10
    raised = tight.validated_update({"history_length": 40})
    again = await R._prepare_messages(7, raised.id, "ещё", topic, storage, raised)
    again_dialog = [item for item in again if item["role"] != "system"]
    assert len(again_dialog) < 30
    assert all(
        not str(item.get("content", "")).startswith("old-0") for item in again_dialog
    )


def test_defaults_do_not_trim_a_short_dialog() -> None:
    from app.infra.settings import get_settings

    settings = get_settings()
    assert int(str(settings.max_history_length)) == 20
    assert settings.image_bucket_capacity == 15
    assert settings.image_bucket_refill_per_min == 0.5
    assert settings.text_rate_limit == 10
    assert settings.image_max_concurrent_per_user == 3
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]
    assert len(trim_messages(messages, 20, max_chars=settings.history_max_chars)) == 2
    assert (
        estimate_tokens(messages, chars_per_token=settings.chars_per_token)
        < settings.turn_input_token_limit
    )


def test_new_env_typo_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.infra.settings import require_valid_settings

    monkeypatch.setenv("TURN_INPUT_TOKEN_LIMIT", "abc")
    with pytest.raises(SystemExit):
        require_valid_settings()


def test_power_user_ids_outside_allowlist_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.infra.settings import get_settings

    monkeypatch.setenv("TELEGRAM_ALLOWED_USER_IDS", "1")
    monkeypatch.setenv("POWER_USER_IDS", "2")
    with pytest.raises(SystemExit):
        _ = get_settings().power_user_ids


def test_needs_input_retry_matches_context_errors() -> None:
    assert _needs_input_retry(AssistAIError("context_length_exceeded", status_code=400))
    assert not _needs_input_retry(
        AssistAIError("context_length_exceeded", status_code=500)
    )
    assert fallback_reason(AssistAIError("big", error_code="INPUT_TOO_LARGE")) is None


@pytest.mark.asyncio
async def test_input_too_large_does_not_call_provider(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: MockRouter,
) -> None:
    _patch_budget(monkeypatch)
    monkeypatch.setenv("TURN_INPUT_TOKEN_LIMIT", "1000")
    route = respx_mock.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, text="data: [DONE]\n\n")
    )
    storage = InMemoryTopicStorage()
    profile = _profile()
    message = _Message()
    await R._process_user_message_locked(
        message,  # type: ignore[arg-type]
        SimpleNamespace(),  # type: ignore[arg-type]
        storage,  # type: ignore[arg-type]
        _Profiles(profile),  # type: ignore[arg-type]
        _State(),  # type: ignore[arg-type]
        user_text="X" * 20_000,
    )
    assert route.call_count == 0
    assert any("слишком большой" in text for text in message.replies)
    topic = await storage.get_active_topic(message.chat.id, profile.id)
    assert topic is not None
    assert await storage.get_history(message.chat.id, profile.id, topic) == []


@pytest.mark.asyncio
async def test_over_limit_history_is_trimmed_before_the_request(
    monkeypatch: pytest.MonkeyPatch,
    assist_client: AssistAIClient,
    respx_mock: MockRouter,
) -> None:
    estimates: list[float] = []
    real = cost_tracker.check_budget

    async def spy(*args: object, **kwargs: object) -> None:
        estimated = kwargs.get("estimated")
        if isinstance(estimated, (int, float)):
            estimates.append(float(estimated))
        return None

    monkeypatch.setattr(R.cost_tracker, "check_budget", spy)
    monkeypatch.setattr(cost_tracker, "check_budget", spy)
    monkeypatch.setattr(cost_tracker, "record", real if False else spy)
    monkeypatch.setenv("TURN_INPUT_TOKEN_LIMIT", "1000")
    route = respx_mock.post(_CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            text=(
                'data: {"choices":[{"index":0,"delta":{"content":"ok"},'
                '"finish_reason":"stop"}]}\n\n'
                "data: [DONE]\n\n"
            ),
        )
    )
    storage = InMemoryTopicStorage(max_length=80)
    profile = _profile(history_length=40)
    topic = await storage.ensure_active(7, profile.id, "Общая")
    for index in range(6):
        await storage.append_message(
            7,
            profile.id,
            topic,
            {"role": "user", "content": "O" * 3000 + str(index)},
            40,
        )
    message = _Message()
    await R._process_user_message_locked(
        message,  # type: ignore[arg-type]
        assist_client,
        storage,  # type: ignore[arg-type]
        _Profiles(profile),  # type: ignore[arg-type]
        _State(),  # type: ignore[arg-type]
        user_text="свежее",
    )
    assert route.call_count == 1
    sent = json.loads(route.calls[0].request.content.decode())["messages"]
    blob = json.dumps(sent, ensure_ascii=False)
    assert "свежее" in blob
    assert "O" * 3000 not in blob
    trimmed_chars = sum(content_chars(item.get("content")) for item in sent)
    expected = cost_tracker.estimate_chat_cost(
        input_chars=max(trimmed_chars, 200),
        output_chars=500,
    )
    assert expected in estimates


@pytest.mark.asyncio
async def test_budget_refusal_mentions_long_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_USER_DAILY_BUDGET_USD", "0.001")
    storage = InMemoryTopicStorage(max_length=80)
    profile = _profile(history_length=40)
    topic = await storage.ensure_active(7, profile.id, "Общая")
    for index in range(20):
        await storage.append_message(
            7,
            profile.id,
            topic,
            {"role": "user", "content": "H" * 3000},
            40,
        )
        await storage.append_message(
            7,
            profile.id,
            topic,
            {"role": "assistant", "content": "A" * 3000},
            40,
        )
    message = _Message()
    await R._process_user_message_locked(
        message,  # type: ignore[arg-type]
        SimpleNamespace(),  # type: ignore[arg-type]
        storage,  # type: ignore[arg-type]
        _Profiles(profile),  # type: ignore[arg-type]
        _State(),  # type: ignore[arg-type]
        user_text="ок",
    )
    assert any("настройках персоны" in text for text in message.replies)


@pytest.mark.asyncio
async def test_context_error_retries_once_and_not_after_a_chunk(
    assist_client: AssistAIClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_budget(monkeypatch)
    calls = {"n": 0}

    async def before_chunk(*_args: object, **_kwargs: object):
        calls["n"] += 1
        if calls["n"] == 1:
            raise AssistAIError("context_length_exceeded", status_code=400)
        yield (
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "дальше"},
                        "finish_reason": "stop",
                    }
                ]
            },
            None,
        )

    monkeypatch.setattr(assist_client, "_iter_chat_with_fallback", before_chunk)
    messages: list[dict[str, Any]] = [{"role": "user", "content": "вопрос"}]
    text = [
        delta.text
        async for delta in assist_client.stream_chat(messages, use_tools=False)
        if delta.kind == "content"
    ]
    assert calls["n"] == 2
    assert "".join(text) == "дальше"

    calls["n"] = 0

    async def two_failures(*_args: object, **_kwargs: object):
        calls["n"] += 1
        raise AssistAIError("maximum context length", status_code=400)
        yield  # pragma: no cover

    monkeypatch.setattr(assist_client, "_iter_chat_with_fallback", two_failures)
    with pytest.raises(AssistAIError):
        _ = [
            delta
            async for delta in assist_client.stream_chat(
                [{"role": "user", "content": "ещё"}], use_tools=False
            )
        ]
    assert calls["n"] == 2

    calls["n"] = 0

    async def after_chunk(*_args: object, **_kwargs: object):
        calls["n"] += 1
        yield (
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "кусок"},
                        "finish_reason": None,
                    }
                ]
            },
            None,
        )
        raise AssistAIError("context window", status_code=400)

    monkeypatch.setattr(assist_client, "_iter_chat_with_fallback", after_chunk)
    with pytest.raises(AssistAIError):
        _ = [
            delta
            async for delta in assist_client.stream_chat(
                [{"role": "user", "content": "ещё"}], use_tools=False
            )
        ]
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_set_history_updates_only_the_button_persona(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = InMemoryAssistantStorage()
    alpha = _profile(id="alpha", name="Альфа")
    beta = _profile(id="beta", name="Бета")
    await storage.save_assistant(7, alpha)
    await storage.save_assistant(7, beta)
    await storage.set_active_assistant(7, alpha.id)
    message = create_autospec(Message, instance=True)
    message.chat = SimpleNamespace(id=7)
    query = SimpleNamespace(
        data=f"set_history:{beta.id}:40",
        message=message,
        answer=AsyncMock(),
    )
    fitted: list[str] = []
    original = M._fit_screen

    def spy(text: str) -> str:
        fitted.append(text)
        return original(text)

    monkeypatch.setattr(M, "_fit_screen", spy)
    monkeypatch.setattr(R, "_edit_or_answer", AsyncMock())
    await on_set_history(
        query,  # type: ignore[arg-type]
        assistant=SimpleNamespace(),  # type: ignore[arg-type]
        assistant_storage=storage,
    )
    saved = {item.id: item for item in await storage.list_assistants(7)}
    assert saved["beta"].history_length == 40
    assert saved["alpha"].history_length is None
    callback = f"set_history:{beta.id}:default"
    assert len(callback.encode("utf-8")) <= 64
    assert fitted
    assert any("12K" in text or "сообщений" in text for text in fitted)


@pytest.mark.asyncio
async def test_power_tier_scales_limits_and_clamps_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TELEGRAM_ALLOWED_USER_IDS", "5")
    monkeypatch.setenv("POWER_USER_IDS", "5")
    monkeypatch.setenv("POWER_LIMITS_MULTIPLIER", "2")
    power = effective_limits(5)
    plain = effective_limits(9)
    assert power.tier == "power"
    assert power.text_limit == 20
    assert power.media_limit == 20
    assert power.capacity == 30
    assert power.refill_per_min == 1
    assert power.concurrency == 6
    assert plain.text_limit == 10
    assert plain.capacity == 15
    assert plain.refill_per_min == 0.5
    limiter = RateLimiter()
    now = 1_000_000.0
    limiter._image_buckets[(5, "image_generate")] = (40.0, now)
    allowed, _retry = await limiter._bucket_memory(
        5,
        "image_generate",
        capacity=power.capacity,
        refill_per_min=0,
        now=now,
        consume=True,
    )
    assert allowed is True
    stored, _ts = limiter._image_buckets[(5, "image_generate")]
    assert stored <= power.capacity
    drained = RateLimiter()
    for _ in range(int(plain.capacity)):
        assert await drained._bucket_memory(
            9,
            "image_generate",
            capacity=plain.capacity,
            refill_per_min=plain.refill_per_min,
            now=now,
            consume=True,
        ) == (True, 0.0)
    _ok, retry = await drained._bucket_memory(
        9,
        "image_generate",
        capacity=plain.capacity,
        refill_per_min=plain.refill_per_min,
        now=now,
        consume=False,
    )
    _ok_power, retry_power = await limiter._bucket_memory(
        5,
        "image_generate",
        capacity=1,
        refill_per_min=power.refill_per_min,
        now=now,
        consume=False,
    )
    # Пустое ведро обычного пользователя ждёт дольше: refill 0.5 против 1.
    assert retry > retry_power


@pytest.mark.asyncio
async def test_production_redis_failure_denies_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")
    limiter = RateLimiter()

    class _Boom:
        async def eval(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("redis down")

    limiter.attach_redis(_Boom())
    assert await limiter.check_text_limit(1) is False
    assert await limiter.consume_image_bucket(1, "image_generate") is False
