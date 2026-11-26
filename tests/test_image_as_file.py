"""Выбор «получить картинку файлом» хранится отдельно у каждой модели."""

from __future__ import annotations

from types import SimpleNamespace

from app.bot.models import _image_as_file
from app.bot.screens.settings import _model_picker_parts, _settings_persona_view
from app.storage.assistant_storage import AssistantProfile

_PROMPT = "Ты отвечаешь коротко и по делу всегда."
_FLARE = "openai/gpt-image-2.6-flare"
_FLUX = "black-forest-labs/flux-3-image-klein"


def _assistant() -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            default_text_model="deepseek/deepseek-v4.1-flash-1017",
            default_image_model=_FLARE,
            default_tts_model="openai/gpt-4o-mini-tts",
        )
    )


def test_file_choice_follows_the_selected_model() -> None:
    profile = AssistantProfile(
        name="Художник",
        system_prompt=_PROMPT,
        image_model=_FLARE,
        image_as_file={_FLARE: True, "нет": False, "": True},
    )
    assistant = _assistant()
    assert _image_as_file(profile, _FLARE) is True
    assert _image_as_file(profile, _FLUX) is False

    text, markup = _model_picker_parts(profile, assistant, "image")  # type: ignore[arg-type]
    labels = [button.text for row in markup.inline_keyboard for button in row]
    data = [button.callback_data for row in markup.inline_keyboard for button in row]
    assert "Получать результат файлом: да" in text
    assert "✅ Файлом" in labels
    assert f"set_file:{profile.id}:0" in data
    assert f"set_file:{profile.id}:1" in data

    other = profile.model_copy(update={"image_model": _FLUX})
    other_text, other_markup = _model_picker_parts(other, assistant, "image")  # type: ignore[arg-type]
    other_labels = [
        button.text for row in other_markup.inline_keyboard for button in row
    ]
    assert "Получать результат файлом: нет" in other_text
    assert "✅ Фото" in other_labels
    persona, _markup = _settings_persona_view(other, assistant)  # type: ignore[arg-type]
    assert "Получать результат файлом: нет" in persona


def test_bad_file_map_does_not_break_an_old_profile() -> None:
    profile = AssistantProfile.model_validate(
        {
            "name": "Старый",
            "system_prompt": _PROMPT,
            "image_as_file": ["flare"],
        }
    )
    assert profile.image_as_file == {}
