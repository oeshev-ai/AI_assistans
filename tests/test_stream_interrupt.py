"""Обрыв стрима оставляет уже полученный текст и не откатывает вопрос."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from respx import MockRouter

from app.bot import runtime as R
from app.bot.constants import MSG_UNAVAILABLE
from app.llm.assist_ai import AssistAIClient
from app.llm.chat_runtime import _LENGTH_MARK, ChatDelta
from app.llm.errors import AssistAIError
from app.storage.assistant_storage import AssistantProfile
from app.storage.topic_history import InMemoryTopicStorage, TopicStorageError


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

    async def answer(self, text: str) -> _Status:
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


def _profile() -> AssistantProfile:
    return AssistantProfile(
        id="persona",
        name="Стандартный",
        system_prompt="Ты тестовый помощник.",
        voice_mode="off",
    )


def _assistant(events: list[ChatDelta | BaseException]) -> SimpleNamespace:
    async def stream_chat(
        messages: list[dict[str, object]],
        model: str | None = None,
    ):
        del messages, model
        for event in events:
            if isinstance(event, BaseException):
                raise event
            yield event

    return SimpleNamespace(
        config=SimpleNamespace(
            default_text_model="openai/gpt-4o",
            default_image_model="openai/gpt-image-1",
        ),
        stream_chat=stream_chat,
    )


def _patch_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _budget(*_args: object, **_kwargs: object) -> None:
        return None

    async def _release(*_args: object, **_kwargs: object) -> float:
        return 0.0

    monkeypatch.setattr(R.cost_tracker, "check_budget", _budget)
    monkeypatch.setattr(R.cost_tracker, "release_reservation", _release)


@pytest.mark.asyncio
async def test_text_interrupt_keeps_partial_answer_and_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_budget(monkeypatch)
    storage = InMemoryTopicStorage()
    profile = _profile()
    message = _Message()
    assistant = _assistant(
        [
            ChatDelta("content", "Первая часть. "),
            ChatDelta("content", "Вторая часть."),
            AssistAIError("поток оборвался"),
        ]
    )
    await R._process_user_message_locked(
        message,  # type: ignore[arg-type]
        assistant,  # type: ignore[arg-type]
        storage,  # type: ignore[arg-type]
        _Profiles(profile),  # type: ignore[arg-type]
        _State(),  # type: ignore[arg-type]
        user_text="нарисуй кота",
    )
    topic_id = await storage.get_active_topic(message.chat.id, profile.id)
    history = await storage.get_history(message.chat.id, profile.id, topic_id)
    assert [item["role"] for item in history] == ["user", "assistant"]
    assert history[0]["content"] == "нарисуй кота"
    assert history[1]["content"] == "Первая часть. Вторая часть."
    assert history[1]["interrupted"] is True
    assert "прерван" not in history[1]["content"]
    assert message.replies == ["Готовлю ответ…"]
    assert message.edits[-1] == "Первая часть. Вторая часть.\n…ответ прерван"
    assert message.status.deleted is False
    assert MSG_UNAVAILABLE not in "".join(message.edits)


@pytest.mark.asyncio
async def test_vision_interrupt_keeps_partial_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_budget(monkeypatch)
    storage = InMemoryTopicStorage()
    profile = _profile()
    message = _Message()
    assistant = _assistant(
        [
            ChatDelta("content", "На фото море"),
            AssistAIError("поток оборвался"),
        ]
    )
    await R._process_vision_message_locked(
        message,  # type: ignore[arg-type]
        assistant,  # type: ignore[arg-type]
        storage,  # type: ignore[arg-type]
        _Profiles(profile),  # type: ignore[arg-type]
        _State(),  # type: ignore[arg-type]
        multimodal_content=[{"type": "text", "text": "что тут"}],
        history_placeholder="фото",
    )
    topic_id = await storage.get_active_topic(message.chat.id, profile.id)
    history = await storage.get_history(message.chat.id, profile.id, topic_id)
    assert history[0]["content"] == "фото"
    assert history[-1]["content"] == "На фото море"
    assert history[-1]["interrupted"] is True
    assert message.edits[-1] == "На фото море\n…ответ прерван"
    assert message.status.deleted is False


@pytest.mark.asyncio
async def test_interrupt_without_text_still_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_budget(monkeypatch)
    storage = InMemoryTopicStorage()
    profile = _profile()
    message = _Message()
    assistant = _assistant([AssistAIError("поток оборвался")])
    await R._process_user_message_locked(
        message,  # type: ignore[arg-type]
        assistant,  # type: ignore[arg-type]
        storage,  # type: ignore[arg-type]
        _Profiles(profile),  # type: ignore[arg-type]
        _State(),  # type: ignore[arg-type]
        user_text="нарисуй кота",
    )
    topic_id = await storage.get_active_topic(message.chat.id, profile.id)
    history = await storage.get_history(message.chat.id, profile.id, topic_id)
    assert history == []
    assert message.edits == [MSG_UNAVAILABLE]
    assert message.status.deleted is False


def test_interrupted_answer_appends_mark_once() -> None:
    assert R._interrupted_answer(["  "]) == ""
    assert R._interrupted_answer(["Раз", ", два"]) == "Раз, два\n…ответ прерван"
    marked = "Раз, два\n…ответ прерван"
    assert R._interrupted_answer([marked]) == marked


_CHAT_URL = "https://api.proxyapi.ru/v1/chat/completions"
_LENGTH_NOTICE = f"\n{_LENGTH_MARK}"


def _sse(events: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for event in events:
        lines.append("data: " + json.dumps(event, ensure_ascii=False))
        lines.append("")
    lines.append("data: [DONE]")
    lines.append("")
    return "\n".join(lines)


def _stream_delta(
    delta: dict[str, Any], finish_reason: str | None = None
) -> dict[str, Any]:
    return {
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ]
    }


def _messages_of(request: httpx.Request) -> list[dict[str, Any]]:
    payload = json.loads(request.content.decode())
    messages = payload["messages"]
    assert isinstance(messages, list)
    return messages


@pytest.mark.asyncio
async def test_length_finish_keeps_mark_out_of_history(
    monkeypatch: pytest.MonkeyPatch,
    assist_client: AssistAIClient,
    respx_mock: MockRouter,
) -> None:
    """finish_reason=length: пометка в чате, в append_messages content чистый."""
    _patch_budget(monkeypatch)
    storage = InMemoryTopicStorage()
    profile = _profile()
    message = _Message()
    appended: list[list[dict[str, Any]]] = []
    original = storage.append_messages

    async def spy(
        chat_id: int,
        assistant_id: str,
        topic_id: str,
        messages: list[dict[str, Any]],
        max_length: int,
    ) -> None:
        appended.append(messages)
        await original(chat_id, assistant_id, topic_id, messages, max_length)

    monkeypatch.setattr(storage, "append_messages", spy)
    respx_mock.post(_CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            text=_sse([_stream_delta({"content": "Фраза обрывает"}, "length")]),
        )
    )
    await R._process_user_message_locked(
        message,  # type: ignore[arg-type]
        assist_client,
        storage,  # type: ignore[arg-type]
        _Profiles(profile),  # type: ignore[arg-type]
        _State(),  # type: ignore[arg-type]
        user_text="продолжай",
    )
    assert appended
    stored = appended[-1][-1]
    assert stored["role"] == "assistant"
    assert stored["content"] == "Фраза обрывает"
    assert _LENGTH_MARK not in stored["content"]
    assert "interrupted" not in stored
    visible = "\n".join([*message.edits, *message.replies])
    assert _LENGTH_MARK in visible


@pytest.mark.asyncio
async def test_error_interrupt_stores_flag_and_clean_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Обрыв по ошибке: в истории interrupted и чистый content, в чате пометка."""
    _patch_budget(monkeypatch)
    storage = InMemoryTopicStorage()
    profile = _profile()
    message = _Message()

    async def fail(*_args: object, **_kwargs: object) -> None:
        raise TopicStorageError("redis")

    monkeypatch.setattr(R, "_remember_model_turn", fail)
    assistant = _assistant(
        [
            ChatDelta("content", "Уже сказал"),
            AssistAIError("поток оборвался"),
        ]
    )
    await R._process_user_message_locked(
        message,  # type: ignore[arg-type]
        assistant,  # type: ignore[arg-type]
        storage,  # type: ignore[arg-type]
        _Profiles(profile),  # type: ignore[arg-type]
        _State(),  # type: ignore[arg-type]
        user_text="вопрос",
    )
    topic_id = await storage.get_active_topic(message.chat.id, profile.id)
    history = await storage.get_history(message.chat.id, profile.id, topic_id)
    assert history[-1]["content"] == "Уже сказал"
    assert history[-1]["interrupted"] is True
    assert R._INTERRUPTED_MARK not in history[-1]["content"]
    assert message.edits[-1] == "Уже сказал\n…ответ прерван"


@pytest.mark.asyncio
async def test_next_turn_payload_drops_interrupted_and_marks(
    monkeypatch: pytest.MonkeyPatch,
    assist_client: AssistAIClient,
    respx_mock: MockRouter,
) -> None:
    """Следующий ход не отправляет флаг interrupted и текст служебных пометок."""
    _patch_budget(monkeypatch)
    storage = InMemoryTopicStorage()
    profile = _profile()
    message = _Message()
    route = respx_mock.post(_CHAT_URL).mock(
        side_effect=[
            httpx.Response(
                200,
                text=_sse([_stream_delta({"content": "Фраза обрывает"}, "length")]),
            ),
            httpx.Response(
                200,
                text=_sse([_stream_delta({"content": "Дальше."}, "stop")]),
            ),
        ]
    )
    await R._process_user_message_locked(
        message,  # type: ignore[arg-type]
        assist_client,
        storage,  # type: ignore[arg-type]
        _Profiles(profile),  # type: ignore[arg-type]
        _State(),  # type: ignore[arg-type]
        user_text="первый",
    )
    topic_id = await storage.get_active_topic(message.chat.id, profile.id)
    seeded = await storage.get_history(message.chat.id, profile.id, topic_id)
    seeded[-1]["content"] = "Фраза обрывает"
    seeded[-1]["interrupted"] = True
    await storage.clear_topic(message.chat.id, profile.id, topic_id)
    for item in seeded:
        await storage.append_message(
            message.chat.id, profile.id, topic_id, item, storage.max_length
        )
    await R._process_user_message_locked(
        message,  # type: ignore[arg-type]
        assist_client,
        storage,  # type: ignore[arg-type]
        _Profiles(profile),  # type: ignore[arg-type]
        _State(),  # type: ignore[arg-type]
        user_text="второй",
    )
    sent = _messages_of(route.calls[-1].request)
    blob = json.dumps(sent, ensure_ascii=False)
    assert all("interrupted" not in item for item in sent)
    assert _LENGTH_MARK not in blob
    assert R._INTERRUPTED_MARK not in blob
    dialog = R._dialog_for_model(
        [
            {
                "role": "assistant",
                "content": "чисто",
                "interrupted": True,
            }
        ]
    )
    assert dialog == [{"role": "assistant", "content": "чисто"}]
