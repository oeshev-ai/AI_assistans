"""Протухший экран пишет в персону с кнопки, а не в текущую активную."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, create_autospec

import pytest
from aiogram.types import Message

from app.bot.constants import _SCREEN_ASSISTANTS, _SCREEN_TOPICS
from app.bot.handlers.assistants import on_assistants_done
from app.bot.handlers.settings import (
    on_set_persona_size,
    on_set_text_model,
    on_set_video,
    on_voice_mode,
    on_voice_name,
)
from app.bot.handlers.topics import (
    on_topic_delete_ask,
    on_topic_new,
    on_topic_rename,
    on_topic_use,
    on_topics_done,
    on_topics_page,
)
from app.bot.screens.assistants import _assistants_keyboard
from app.bot.screens.settings import _model_picker_parts
from app.bot.screens.topics import _topics_keyboard
from app.llm.models_config import AVAILABLE_IMAGE_MODELS, AVAILABLE_TEXT_MODELS
from app.storage.assistant_storage import (
    AssistantProfile,
    InMemoryAssistantStorage,
    default_assistant_profile,
)
from app.storage.topic_history import InMemoryTopicStorage

_PROMPT = "Ты отвечаешь коротко и по делу всегда."


class _State:
    def __init__(self) -> None:
        self.data: dict[str, Any] = {}
        self.current: Any = None

    async def get_data(self) -> dict[str, Any]:
        return dict(self.data)

    async def set_data(self, data: dict[str, Any]) -> None:
        self.data = dict(data)

    async def set_state(self, state: Any) -> None:
        self.current = state

    async def update_data(
        self, payload: dict[str, Any] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        if payload:
            kwargs.update(payload)
        self.data.update(kwargs)
        return dict(self.data)


def _assistant() -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            default_text_model=AVAILABLE_TEXT_MODELS[0],
            default_image_model=AVAILABLE_IMAGE_MODELS[0],
            default_tts_model="openai/gpt-4o-mini-tts",
            default_tts_voice="alloy",
            default_tts_format="mp3",
            speechkit_role="friendly",
        )
    )


def _profile(profile_id: str, name: str) -> AssistantProfile:
    return default_assistant_profile().model_copy(
        update={
            "id": profile_id,
            "name": name,
            "image_model": AVAILABLE_IMAGE_MODELS[0],
        }
    )


def _query(data: str, chat_id: int = 1) -> SimpleNamespace:
    message = create_autospec(Message, instance=True)
    message.chat = SimpleNamespace(id=chat_id)
    message.message_id = 10
    message.answer = AsyncMock()
    return SimpleNamespace(data=data, message=message, answer=AsyncMock())


def _callbacks(markup: Any) -> list[str]:
    return [
        button.callback_data
        for row in markup.inline_keyboard
        for button in row
        if button.callback_data
    ]


def test_topic_and_settings_callbacks_fit_64_bytes() -> None:
    assistant_id = "a" * 32
    topics = [{"id": "abcd1234", "name": f"Тема {index}"} for index in range(11)]
    topic_markup = _topics_keyboard(topics, "abcd1234", assistant_id=assistant_id)
    topic_data = _callbacks(topic_markup)
    assert f"topic:new:{assistant_id}" in topic_data
    assert f"topic:use:{assistant_id}:abcd1234" in topic_data
    assert f"topic_page:{assistant_id}:1" in topic_data
    assert topic_data[-1] == "topic:done"
    assistants = _assistants_keyboard(
        [AssistantProfile(id="p", name="Альфа", system_prompt=_PROMPT)], "p"
    )
    assert _callbacks(assistants)[-1] == "asst:done"

    profile = AssistantProfile(
        id=assistant_id,
        name="Длинная",
        system_prompt=_PROMPT,
        image_model=AVAILABLE_IMAGE_MODELS[0],
        tts_model="yandex/speechkit",
        tts_voice="marina",
    )
    assistant = _assistant()
    settings_data: list[str] = []
    for kind in ("text", "image", "video", "tts"):
        _text, markup = _model_picker_parts(profile, assistant, kind)  # type: ignore[arg-type]
        settings_data.extend(_callbacks(markup))
    assert f"set_text:{assistant_id}:0" in settings_data
    assert f"set_size:{assistant_id}:1792x1024" in settings_data
    assert f"set_bg:{assistant_id}:transparent" in settings_data
    assert f"video:{assistant_id}:aspect:21:9" in settings_data
    assert f"voice_mode:{assistant_id}:always" in settings_data
    assert f"voice_role:{assistant_id}:friendly" in settings_data
    assert f"picker:done:{assistant_id}" in settings_data
    for item in (*topic_data, *settings_data):
        assert len(item.encode("utf-8")) <= 64


@pytest.mark.asyncio
async def test_topic_new_and_use_keep_persona_from_the_button(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.bot.runtime as runtime

    topics = InMemoryTopicStorage()
    chat_id = 4
    topic_id = await topics.create_topic(chat_id, "personaA", "Alpha")
    await topics.create_topic(chat_id, "personaB", "Beta")
    monkeypatch.setattr(
        runtime, "_persona_id_for_query", AsyncMock(return_value="personaB")
    )
    monkeypatch.setattr(runtime, "_menu_for", AsyncMock(return_value=SimpleNamespace()))
    monkeypatch.setattr(runtime, "_close_screen", AsyncMock())
    monkeypatch.setattr(runtime, "_apply_reply_menu", AsyncMock())

    state = _State()
    new_query = _query("topic:new:personaA", chat_id=chat_id)
    await on_topic_new(
        new_query,  # type: ignore[arg-type]
        assistant_storage=SimpleNamespace(),
        state=state,  # type: ignore[arg-type]
    )
    assert state.data["flow_assistant_id"] == "personaA"

    use_query = _query(f"topic:use:personaA:{topic_id}", chat_id=chat_id)
    await on_topic_use(
        use_query,  # type: ignore[arg-type]
        topic_storage=topics,
        assistant_storage=SimpleNamespace(),
        state=_State(),  # type: ignore[arg-type]
    )
    assert await topics.get_active_topic(chat_id, "personaA") == topic_id
    assert await topics.get_active_topic(chat_id, "personaB") == ""

    state = _State()
    rename_query = _query(f"topic:rename:personaA:{topic_id}", chat_id=chat_id)
    await on_topic_rename(
        rename_query,  # type: ignore[arg-type]
        topic_storage=topics,
        assistant_storage=SimpleNamespace(),
        state=state,  # type: ignore[arg-type]
    )
    assert state.data["flow_assistant_id"] == "personaA"
    assert state.data["flow_topic_id"] == topic_id

    seen: dict[str, Any] = {}

    async def edit(_message: Any, _text: str, markup: Any) -> None:
        seen["markup"] = markup

    monkeypatch.setattr(runtime, "_edit_inline", edit)
    delete_query = _query(f"topic_del_ask:personaA:{topic_id}", chat_id=chat_id)
    await on_topic_delete_ask(
        delete_query,  # type: ignore[arg-type]
        topic_storage=topics,
        assistant_storage=SimpleNamespace(),
    )
    confirm = _callbacks(seen["markup"])
    assert f"topic_del_yes:personaA:{topic_id}" in confirm
    assert "topic:list:personaA" in confirm


@pytest.mark.asyncio
async def test_topics_page_lists_persona_from_the_button(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.bot.runtime as runtime

    topics = InMemoryTopicStorage()
    chat_id = 5
    topic_id = await topics.create_topic(chat_id, "personaA", "Alpha")
    monkeypatch.setattr(
        runtime, "_persona_id_for_query", AsyncMock(return_value="personaB")
    )
    seen: dict[str, Any] = {}

    async def edit(_query: Any, text: str, markup: Any) -> None:
        seen["text"] = text
        seen["markup"] = markup

    monkeypatch.setattr(runtime, "_edit_or_answer", edit)
    query = _query("topic_page:personaA:0", chat_id=chat_id)
    await on_topics_page(
        query,  # type: ignore[arg-type]
        topic_storage=topics,
        assistant_storage=SimpleNamespace(),
    )
    assert f"topic:use:personaA:{topic_id}" in _callbacks(seen["markup"])
    assert "Alpha" in seen["text"] or any(
        "Alpha" in button.text
        for row in seen["markup"].inline_keyboard
        for button in row
    )


@pytest.mark.asyncio
async def test_settings_buttons_update_persona_from_the_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.bot.runtime as runtime
    import app.bot.screens as screens

    storage = InMemoryAssistantStorage()
    chat_id = 6
    alpha = _profile("alpha", "Альфа")
    beta = _profile("beta", "Бета")
    await storage.save_assistant(chat_id, alpha)
    await storage.save_assistant(chat_id, beta)
    await storage.set_active_assistant(chat_id, alpha.id)
    assistant = _assistant()
    monkeypatch.setattr(screens, "_model_picker_parts", lambda *_a, **_k: ("t", None))
    monkeypatch.setattr(runtime, "_edit_or_answer", AsyncMock())
    monkeypatch.setattr(runtime, "_return_to_settings", AsyncMock())

    size_query = _query(f"set_size:{beta.id}:1792x1024", chat_id=chat_id)
    await on_set_persona_size(
        size_query,  # type: ignore[arg-type]
        assistant=assistant,  # type: ignore[arg-type]
        assistant_storage=storage,
    )
    video_query = _query(f"video:{beta.id}:aspect:9:16", chat_id=chat_id)
    await on_set_video(
        video_query,  # type: ignore[arg-type]
        assistant=assistant,  # type: ignore[arg-type]
        assistant_storage=storage,
    )
    voice_query = _query(f"voice_mode:{beta.id}:off", chat_id=chat_id)
    await on_voice_mode(
        voice_query,  # type: ignore[arg-type]
        assistant=assistant,  # type: ignore[arg-type]
        assistant_storage=storage,
    )
    text_query = _query(f"set_text:{beta.id}:0", chat_id=chat_id)
    await on_set_text_model(
        text_query,  # type: ignore[arg-type]
        assistant=assistant,  # type: ignore[arg-type]
        assistant_storage=storage,
        state=_State(),  # type: ignore[arg-type]
    )

    saved = {item.id: item for item in await storage.list_assistants(chat_id)}
    assert saved[beta.id].image_size == "1792x1024"
    assert saved[beta.id].video_aspect_ratio == "9:16"
    assert saved[beta.id].voice_mode == "off"
    assert saved[beta.id].text_model == AVAILABLE_TEXT_MODELS[0]
    assert saved[alpha.id].image_size is None
    assert saved[alpha.id].video_aspect_ratio is None
    assert saved[alpha.id].voice_mode == "auto"
    assert saved[alpha.id].text_model is None
    active = await storage.get_active_assistant(chat_id)
    assert active is not None
    assert active.id == alpha.id


@pytest.mark.asyncio
async def test_done_closes_the_list_without_moving_the_menu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.bot.runtime as runtime

    monkeypatch.setattr(runtime, "_close_screen", AsyncMock())
    monkeypatch.setattr(runtime, "_apply_reply_menu", AsyncMock())
    state = _State()

    assistants = _query("asst:done")
    await on_assistants_done(assistants, state)  # type: ignore[arg-type]
    runtime._close_screen.assert_awaited_once()
    assert runtime._close_screen.await_args.args[2] == _SCREEN_ASSISTANTS
    runtime._apply_reply_menu.assert_not_called()

    runtime._close_screen.reset_mock()
    topics = _query("topic:done")
    await on_topics_done(topics, state)  # type: ignore[arg-type]
    assert runtime._close_screen.await_args.args[2] == _SCREEN_TOPICS
    runtime._apply_reply_menu.assert_not_called()


@pytest.mark.asyncio
async def test_voice_name_storage_error_is_not_an_unknown_voice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.bot.runtime as runtime
    import app.bot.screens as screens

    storage = InMemoryAssistantStorage()
    chat_id = 8
    profile = _profile("voice", "Голос")
    await storage.save_assistant(chat_id, profile)

    async def boom(_chat_id: int, _profile: AssistantProfile) -> None:
        raise ValueError("сбой записи")

    monkeypatch.setattr(storage, "save_assistant", boom)
    monkeypatch.setattr(screens, "_model_picker_parts", lambda *_a, **_k: ("t", None))
    monkeypatch.setattr(runtime, "_edit_or_answer", AsyncMock())
    query = _query(f"voice_name:{profile.id}:0", chat_id=chat_id)
    with pytest.raises(ValueError, match="сбой записи"):
        await on_voice_name(
            query,  # type: ignore[arg-type]
            assistant=_assistant(),  # type: ignore[arg-type]
            assistant_storage=storage,
        )
    query.answer.assert_not_awaited()

    bad = _query(f"voice_name:{profile.id}:99", chat_id=chat_id)
    await on_voice_name(
        bad,  # type: ignore[arg-type]
        assistant=_assistant(),  # type: ignore[arg-type]
        assistant_storage=storage,
    )
    bad.answer.assert_awaited_once_with("Неизвестный голос", show_alert=True)
