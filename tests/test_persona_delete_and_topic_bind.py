"""Удаление персоны: профиль раньше тем; темы wizard помнят assistant_id."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from app.bot.handlers.assistants import on_asst_del_yes
from app.bot.handlers.topics import on_topic_delete_yes, sg_rename_topic_name
from app.storage.assistant_storage import (
    DEFAULT_ASSISTANT_ID,
    InMemoryAssistantStorage,
    default_assistant_profile,
)
from app.storage.topic_history import InMemoryTopicStorage


class _Query:
    def __init__(self, data: str, chat_id: int = 1) -> None:
        self.data = data
        self.message = SimpleNamespace(
            chat=SimpleNamespace(id=chat_id),
            message_id=10,
            answer=AsyncMock(),
        )
        self.answer = AsyncMock()
        self.from_user = SimpleNamespace(id=2)


@pytest.mark.asyncio
async def test_delete_persona_keeps_topics_if_profile_delete_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.bot.handlers.assistants as handlers
    import app.bot.runtime as runtime

    assistants = InMemoryAssistantStorage()
    topics = InMemoryTopicStorage()
    chat_id = 3
    profile = default_assistant_profile().model_copy(
        update={"id": "custom1", "name": "Тест"}
    )
    await assistants.save_assistant(chat_id, profile)
    await topics.create_topic(chat_id, "custom1", "Работа")

    async def boom(_chat_id: int, _profile_id: str) -> None:
        raise handlers.AssistantStorageError("fail")

    monkeypatch.setattr(assistants, "delete_assistant", boom)
    monkeypatch.setattr(runtime, "_query_message", lambda q: q.message)
    monkeypatch.setattr(handlers.screens, "_settings_hub_view", lambda *_a: ("t", None))

    query = _Query("asst:del_yes:custom1", chat_id=chat_id)
    await on_asst_del_yes(
        query,  # type: ignore[arg-type]
        assistant=SimpleNamespace(),
        assistant_storage=assistants,
        topic_storage=topics,
        state=SimpleNamespace(update_data=AsyncMock()),  # type: ignore[arg-type]
    )
    listed = await topics.list_topics(chat_id, "custom1")
    assert any(item["name"] == "Работа" for item in listed)
    query.answer.assert_awaited()
    assert "fail" in query.answer.await_args.args[0]


@pytest.mark.asyncio
async def test_delete_persona_profile_first_then_topics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.bot.handlers.assistants as handlers
    import app.bot.runtime as runtime

    assistants = InMemoryAssistantStorage()
    topics = InMemoryTopicStorage()
    chat_id = 4
    profile = default_assistant_profile().model_copy(
        update={"id": "gone", "name": "Удалить"}
    )
    await assistants.save_assistant(chat_id, profile)
    await topics.create_topic(chat_id, "gone", "Старая")
    order: list[str] = []

    real_delete = assistants.delete_assistant

    async def track_delete(cid: int, pid: str) -> None:
        order.append("profile")
        await real_delete(cid, pid)

    real_topics = topics.delete_assistant_topics

    async def track_topics(cid: int, aid: str) -> None:
        order.append("topics")
        await real_topics(cid, aid)

    monkeypatch.setattr(assistants, "delete_assistant", track_delete)
    monkeypatch.setattr(topics, "delete_assistant_topics", track_topics)
    monkeypatch.setattr(runtime, "_query_message", lambda q: q.message)
    monkeypatch.setattr(runtime, "_active_profile", AsyncMock(return_value=default_assistant_profile()))
    monkeypatch.setattr(runtime, "_menu_for", AsyncMock(return_value=SimpleNamespace()))
    monkeypatch.setattr(runtime, "_edit_inline", AsyncMock())
    monkeypatch.setattr(handlers.screens, "_settings_hub_view", lambda *_a: ("t", None))

    query = _Query("asst:del_yes:gone", chat_id=chat_id)
    message = query.message
    message.answer = AsyncMock()
    await on_asst_del_yes(
        query,  # type: ignore[arg-type]
        assistant=SimpleNamespace(),
        assistant_storage=assistants,
        topic_storage=topics,
        state=SimpleNamespace(update_data=AsyncMock()),  # type: ignore[arg-type]
    )
    assert order == ["profile", "topics"]
    assert await topics.list_topics(chat_id, "gone") == []


@pytest.mark.asyncio
async def test_topic_delete_uses_assistant_from_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import create_autospec

    from aiogram.types import Message

    import app.bot.runtime as runtime

    called: dict[str, str] = {}

    async def finish(
        chat_id: int,
        assistant_id: str,
        topic_id: str,
        *_args: Any,
        **_kwargs: Any,
    ) -> tuple[str, Any, Any]:
        called["assistant_id"] = assistant_id
        called["topic_id"] = topic_id
        return "text", SimpleNamespace(), SimpleNamespace()

    monkeypatch.setattr(runtime, "_finish_topic_delete", finish)
    monkeypatch.setattr(runtime, "_edit_inline", AsyncMock())
    monkeypatch.setattr(runtime, "_apply_reply_menu", AsyncMock())
    monkeypatch.setattr(
        runtime,
        "_persona_id_for_query",
        AsyncMock(return_value="WRONG"),
    )

    message = create_autospec(Message, instance=True)
    message.chat = SimpleNamespace(id=1)
    message.message_id = 10
    query = SimpleNamespace(
        data="topic_del_yes:personaA:topic99",
        message=message,
        answer=AsyncMock(),
    )
    await on_topic_delete_yes(
        query,  # type: ignore[arg-type]
        topic_storage=SimpleNamespace(),
        assistant_storage=SimpleNamespace(),
        state=SimpleNamespace(update_data=AsyncMock()),  # type: ignore[arg-type]
    )
    assert called == {"assistant_id": "personaA", "topic_id": "topic99"}


@pytest.mark.asyncio
async def test_rename_uses_flow_assistant_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.bot.runtime as runtime

    topics = InMemoryTopicStorage()
    await topics.create_topic(1, "bound", "Alpha")
    topic_id = (await topics.list_topics(1, "bound"))[0]["id"]

    class _State:
        def __init__(self) -> None:
            self.data = {
                "flow_topic_id": topic_id,
                "flow_assistant_id": "bound",
            }

        async def get_data(self) -> dict[str, Any]:
            return dict(self.data)

    monkeypatch.setattr(runtime, "_reset_flow", AsyncMock())
    monkeypatch.setattr(runtime, "_menu_for", AsyncMock(return_value=SimpleNamespace()))
    monkeypatch.setattr(runtime, "_apply_reply_menu", AsyncMock())
    monkeypatch.setattr(runtime, "_open_topics", AsyncMock())
    monkeypatch.setattr(
        runtime,
        "_active_profile",
        AsyncMock(
            return_value=SimpleNamespace(id=DEFAULT_ASSISTANT_ID),
        ),
    )

    answers: list[str] = []

    class _Message:
        chat = SimpleNamespace(id=1)
        text = "Beta"

        async def answer(self, text: str, **_kwargs: Any) -> None:
            answers.append(text)

    await sg_rename_topic_name(
        _Message(),  # type: ignore[arg-type]
        state=_State(),  # type: ignore[arg-type]
        topic_storage=topics,
        assistant_storage=SimpleNamespace(),
    )
    renamed = await topics.list_topics(1, "bound")
    assert renamed[0]["name"] == "Beta"
    # Активная «стандартная» персона не должна получить эту тему.
    assert await topics.list_topics(1, DEFAULT_ASSISTANT_ID) == []
