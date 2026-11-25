"""Tool-дельта меняет статус чата до того, как поток продолжится."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.bot import runtime as R
from app.llm.chat_runtime import ChatDelta
from app.storage.assistant_storage import AssistantProfile


class _Chat:
    id = 7


class _User:
    id = 11


class _Status:
    def __init__(self, edits: list[str]) -> None:
        self.edits = edits
        self.message_id = 1

    async def edit_text(self, text: str) -> None:
        self.edits.append(text)


class _Message:
    def __init__(self, edits: list[str]) -> None:
        self.chat = _Chat()
        self.from_user = _User()
        self.bot = None
        self.edits = edits
        self.replies: list[str] = []

    async def answer(self, text: str) -> _Status:
        self.replies.append(text)
        return _Status(self.edits)


def _profile() -> AssistantProfile:
    return AssistantProfile(
        name="Стандартный",
        system_prompt="Ты тестовый помощник.",
        voice_mode="off",
    )


def _assistant(edits: list[str], *tool_texts: str) -> SimpleNamespace:
    resumed_with_status: list[bool] = []

    async def stream_chat(
        messages: list[dict[str, object]],
        model: str | None = None,
    ):
        del messages, model
        for text in tool_texts:
            yield ChatDelta("tool", text)
            resumed_with_status.append(
                bool(edits)
                and edits[-1].startswith("🎨 ")
                and text.split()[0] in edits[-1]
            )
        yield ChatDelta("content", "Готово")

    client = SimpleNamespace(
        config=SimpleNamespace(
            default_text_model="openai/gpt-4o",
            default_image_model="openai/gpt-image-1",
        ),
        stream_chat=stream_chat,
        resumed_with_status=resumed_with_status,
    )
    return client


def _patch_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _budget(*_args: object, **_kwargs: object) -> None:
        return None

    async def _release(*_args: object, **_kwargs: object) -> float:
        return 0.0

    async def _profile_of(*_args: object, **_kwargs: object) -> AssistantProfile:
        return _profile()

    async def _topic(*_args: object, **_kwargs: object) -> str:
        return "topic"

    async def _history(*_args: object, **_kwargs: object) -> list[dict[str, str]]:
        return [{"role": "user", "content": "нарисуй"}]

    async def _noop(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(R.cost_tracker, "check_budget", _budget)
    monkeypatch.setattr(R.cost_tracker, "release_reservation", _release)
    monkeypatch.setattr(R, "_active_profile", _profile_of)
    monkeypatch.setattr(R, "_active_topic", _topic)
    monkeypatch.setattr(R, "_prepare_messages", _history)
    monkeypatch.setattr(R, "_prepare_messages_with_image", _history)
    monkeypatch.setattr(R, "_remember_model_turn", _noop)
    monkeypatch.setattr(R, "_send_extracted_images", _noop)
    monkeypatch.setattr(R, "_send_outbound_media", _noop)


@pytest.mark.asyncio
async def test_text_turn_shows_tool_status_before_stream_resumes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_turn(monkeypatch)
    edits: list[str] = []
    message = _Message(edits)
    assistant = _assistant(
        edits,
        'generate_image {"prompt": "кот"}',
        'generate_video {"prompt": "кот бежит"}',
    )
    await R._process_user_message_locked(
        message,  # type: ignore[arg-type]
        assistant,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        user_text="нарисуй кота",
    )
    assert message.replies == ["Готовлю ответ…"]
    assert edits[0] == "🎨 generate_image…"
    assert edits[1] == "🎨 generate_image, generate_video…"
    assert edits[-1] == "Готово"
    assert assistant.resumed_with_status == [True, True]


@pytest.mark.asyncio
async def test_vision_turn_shows_tool_status_before_stream_resumes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_turn(monkeypatch)
    edits: list[str] = []
    message = _Message(edits)
    assistant = _assistant(edits, 'generate_video {"prompt": "панорама"}')
    await R._process_vision_message_locked(
        message,  # type: ignore[arg-type]
        assistant,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        multimodal_content=[{"type": "text", "text": "опиши"}],
        history_placeholder="фото",
        status_text="Анализирую изображение…",
    )
    assert message.replies == ["Анализирую изображение…"]
    assert edits[0] == "🎨 generate_video…"
    assert edits[-1] == "Готово"
    assert assistant.resumed_with_status == [True]


def test_note_tool_progress_uses_tool_name_only() -> None:
    names: list[str] = []
    assert R._note_tool_progress(names, 'generate_image {"prompt": "кот"}') == (
        "🎨 generate_image…"
    )
    assert names == ["generate_image"]
    assert R._note_tool_progress(names, "generate_image ещё раз") == (
        "🎨 generate_image…"
    )
