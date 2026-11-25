"""Обрыв стрима оставляет уже полученный текст и не откатывает вопрос."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.bot import runtime as R
from app.bot.constants import MSG_UNAVAILABLE
from app.llm.chat_runtime import ChatDelta
from app.llm.errors import AssistAIError
from app.storage.assistant_storage import AssistantProfile
from app.storage.topic_history import InMemoryTopicStorage


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
    assert history[1]["content"] == "Первая часть. Вторая часть.\n…ответ прерван"
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
    assert history[-1]["content"] == "На фото море\n…ответ прерван"
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
