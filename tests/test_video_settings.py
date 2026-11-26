"""Настройки ролика персоны: хранение, экран и передача в generate_video."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.bot.constants import TELEGRAM_TEXT_SAFE
from app.bot.models import _fit_screen
from app.bot.screens.settings import _model_picker_parts, _settings_persona_view
from app.bot.telegram_text import telegram_text_units
from app.llm.media_api import VideoResult
from app.llm.video_settings import (
    VIDEO_ASPECT_RATIOS,
    VIDEO_DURATIONS,
    VIDEO_RESOLUTIONS,
    VideoSettings,
    VideoSettingsScope,
    parse_video_callback,
    split_video_persona,
    video_settings_from,
)
from app.media.media_tools import GenerateVideoTool
from app.storage.assistant_storage import (
    DEFAULT_SYSTEM_PROMPT,
    AssistantProfile,
    InMemoryAssistantStorage,
    default_assistant_profile,
)

_PROMPT = "Ты отвечаешь коротко и по делу всегда."


def _assistant() -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            default_text_model="deepseek/deepseek-v4.1-flash-1017",
            default_image_model="openai/gpt-image-2.6-flare",
            default_tts_model="openai/gpt-4o-mini-tts",
        )
    )


class _VideoClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def generate_video(self, prompt: str, **kwargs: object) -> VideoResult:
        self.calls.append({"prompt": prompt, **kwargs})
        return VideoResult(data=b"mp4", filename="video.mp4")


def test_empty_profile_uses_seedance_defaults() -> None:
    settings = video_settings_from(default_assistant_profile())
    assert settings == VideoSettings("16:9", 5, "720p", False)


def test_invalid_video_fields_fall_back_to_defaults() -> None:
    profile = AssistantProfile(
        name="Мусор",
        system_prompt=_PROMPT,
        video_aspect_ratio="99:1",
        video_duration=99,
        video_resolution="4k",
        video_generate_audio="maybe",
    )
    assert profile.video_aspect_ratio is None
    assert profile.video_duration is None
    assert profile.video_resolution is None
    assert profile.video_generate_audio is None
    assert video_settings_from(profile) == VideoSettings("16:9", 5, "720p", False)


def test_generate_audio_string_is_stored_as_bool() -> None:
    profile = AssistantProfile(
        name="Звук",
        system_prompt=_PROMPT,
        video_generate_audio="true",
    )
    assert profile.video_generate_audio is True


def test_system_prompt_points_at_persona_video_settings() -> None:
    assert "настроек персоны" in DEFAULT_SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_each_persona_keeps_its_own_video_settings() -> None:
    storage = InMemoryAssistantStorage()
    try:
        vertical = AssistantProfile(
            name="Вертикаль",
            system_prompt=_PROMPT,
            video_aspect_ratio="9:16",
            video_duration=8,
            video_resolution="480p",
            video_generate_audio=True,
        )
        wide = AssistantProfile(
            name="Широкий",
            system_prompt=_PROMPT,
            video_aspect_ratio="21:9",
            video_duration=15,
            video_resolution="720p",
            video_generate_audio=False,
        )
        await storage.save_assistant(1, vertical)
        await storage.save_assistant(1, wide)
        listed = {item.name: item for item in await storage.list_assistants(1)}
        assert listed["Вертикаль"].video_aspect_ratio == "9:16"
        assert listed["Вертикаль"].video_duration == 8
        assert listed["Вертикаль"].video_resolution == "480p"
        assert listed["Вертикаль"].video_generate_audio is True
        assert listed["Широкий"].video_aspect_ratio == "21:9"
        assert listed["Широкий"].video_duration == 15
        assert listed["Широкий"].video_generate_audio is False
    finally:
        await storage.close()


def test_persona_menu_puts_video_under_image_model() -> None:
    profile = AssistantProfile(
        name="Кино",
        system_prompt=_PROMPT,
        video_aspect_ratio="9:16",
        video_duration=10,
        video_resolution="480p",
        video_generate_audio=True,
    )
    text, markup = _settings_persona_view(profile, _assistant())  # type: ignore[arg-type]
    buttons = [button for row in markup.inline_keyboard for button in row]
    labels = [button.text for button in buttons]
    callbacks = [button.callback_data for button in buttons]
    image_at = labels.index("Модель изображений")
    assert labels[image_at + 1] == "Видео"
    assert callbacks[callbacks.index("set_image_menu") + 1] == "video_menu"
    assert "Видео: 9:16, 10 с, 480p, со звуком" in text


def test_settings_screens_fit_telegram_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """Длинный каталог или подсказка обрезаются, кнопки экрана остаются."""
    import app.bot.screens.settings as settings

    calls: list[str] = []
    real_fit = _fit_screen

    def spy(text: str) -> str:
        calls.append(text)
        return real_fit(text)

    monkeypatch.setattr(settings.M, "_fit_screen", spy)
    monkeypatch.setattr(settings.M, "_hints_block", lambda *_a, **_k: "я" * 5000)
    monkeypatch.setattr(settings, "model_hint", lambda _model: "я" * 5000)
    profile = AssistantProfile(name="Кино", system_prompt=_PROMPT)
    assistant = SimpleNamespace(
        config=SimpleNamespace(
            default_text_model="deepseek/deepseek-v4.1-flash-1017",
            default_image_model="openai/gpt-image-2.6-flare",
            default_tts_model="openai/gpt-4o-mini-tts",
            default_tts_voice="alloy",
            default_tts_format="opus",
        )
    )
    screens = [
        _model_picker_parts(profile, assistant, "text"),  # type: ignore[arg-type]
        _model_picker_parts(profile, assistant, "image"),  # type: ignore[arg-type]
        _model_picker_parts(profile, assistant, "video"),  # type: ignore[arg-type]
        _model_picker_parts(profile, assistant, "tts"),  # type: ignore[arg-type]
        _settings_persona_view(profile, assistant),  # type: ignore[arg-type]
    ]
    assert len(calls) == len(screens)
    for text, markup in screens:
        assert telegram_text_units(text) <= TELEGRAM_TEXT_SAFE
        assert markup.inline_keyboard
    assert screens[0][0].endswith("…")
    assert screens[1][0].endswith("…")
    assert screens[4][0].endswith("…")
    assert "Видео — «Кино»" in screens[2][0]
    assert "Голосовой ответ — «Кино»" in screens[3][0]


def test_video_picker_offers_seedance_choices() -> None:
    profile = default_assistant_profile()
    text, markup = _model_picker_parts(
        profile,
        _assistant(),  # type: ignore[arg-type]
        "video",
    )
    buttons = [button for row in markup.inline_keyboard for button in row]
    data = [button.callback_data for button in buttons]
    labels = [button.text for button in buttons]
    for ratio in VIDEO_ASPECT_RATIOS:
        assert f"video:{profile.id}:aspect:{ratio}" in data
    for seconds in VIDEO_DURATIONS:
        assert f"video:{profile.id}:dur:{seconds}" in data
    for resolution in VIDEO_RESOLUTIONS:
        assert f"video:{profile.id}:res:{resolution}" in data
    assert f"video:{profile.id}:audio:0" in data
    assert f"video:{profile.id}:audio:1" in data
    assert f"picker:done:{profile.id}" in data
    assert "✅ 16:9" in labels
    assert "✅ 5с" in labels
    assert "✅ 720p" in labels
    assert "✅ Без звука" in labels
    assert "Формат: 16:9" in text
    assert "Длительность: 5 с" in text
    assert "Разрешение: 720p" in text
    assert "Звук: выкл" in text


def test_parse_video_callback_rejects_unknown_values() -> None:
    assert parse_video_callback("video:aspect:9:16") == (
        "video_aspect_ratio",
        "9:16",
        "Формат: 9:16",
    )
    assert parse_video_callback("video:dur:12") == (
        "video_duration",
        12,
        "Длительность: 12 с",
    )
    assert parse_video_callback("video:res:480p") == (
        "video_resolution",
        "480p",
        "Разрешение: 480p",
    )
    assert parse_video_callback("video:audio:1") == (
        "video_generate_audio",
        True,
        "Звук: вкл",
    )
    assert parse_video_callback("video:aspect:99:1") is None
    assert parse_video_callback("video:dur:3") is None
    assert parse_video_callback("video:res:1080p") is None
    assert parse_video_callback("video:audio:maybe") is None
    assert split_video_persona("video:aspect:9:16") == (None, "video:aspect:9:16")
    assert split_video_persona("video:personaA:aspect:21:9") == (
        "personaA",
        "video:aspect:21:9",
    )
    parsed = parse_video_callback(split_video_persona("video:personaA:aspect:21:9")[1])
    assert parsed is not None
    assert parsed[1] == "21:9"


@pytest.mark.asyncio
async def test_tool_sends_persona_settings_not_model_args() -> None:
    client = _VideoClient()
    tool = GenerateVideoTool(client)  # type: ignore[arg-type]
    assert "duration" not in tool.parameters["properties"]
    assert "resolution" not in tool.parameters["properties"]
    settings = VideoSettings("9:16", 10, "480p", True)
    with VideoSettingsScope(settings):
        payload = json.loads(
            await tool.execute(prompt="сцена", duration=15, resolution="720p")
        )
    assert payload["status"] == "success"
    sent = client.calls[0]
    assert sent["aspect_ratio"] == "9:16"
    assert sent["duration"] == 10
    assert sent["resolution"] == "480p"
    assert sent["generate_audio"] is True


@pytest.mark.asyncio
async def test_tool_without_persona_scope_keeps_explicit_duration() -> None:
    client = _VideoClient()
    tool = GenerateVideoTool(client)  # type: ignore[arg-type]
    payload = json.loads(
        await tool.execute(prompt="сцена", duration=8, resolution="720p")
    )
    assert payload["status"] == "success"
    sent = client.calls[0]
    assert sent["duration"] == 8
    assert sent["resolution"] == "720p"
    assert "aspect_ratio" not in sent
    assert "generate_audio" not in sent
