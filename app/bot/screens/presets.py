"""Screen builders."""

from __future__ import annotations

from app.bot import models as M
from app.bot.constants import (
    _VOICE_MODE_LABELS,
    ASSISTANT_PRESETS,
    PAGE_SIZE,
    TELEGRAM_TEXT_LIMIT,
    AssistAIClient,
    AssistantPreset,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)


def _presets_text(page: int = 0) -> str:
    """Текст каталога персон со страницей."""
    current, total = M._page_bounds(len(ASSISTANT_PRESETS), page)
    chunk = ASSISTANT_PRESETS[current * PAGE_SIZE : (current + 1) * PAGE_SIZE]
    lines = ["📚 Каталог персон", ""]
    for preset in chunk:
        lines.append(f"{preset.icon} {preset.name} — {preset.summary}")
    lines.append("")
    lines.append(f"Стр. {current + 1}/{total}")
    return "\n".join(lines)


def _presets_keyboard(page: int = 0) -> InlineKeyboardMarkup:
    """Кнопки пресетов текущей страницы."""
    current, total = M._page_bounds(len(ASSISTANT_PRESETS), page)
    chunk = ASSISTANT_PRESETS[current * PAGE_SIZE : (current + 1) * PAGE_SIZE]
    rows: list[list[InlineKeyboardButton]] = []
    for preset in chunk:
        rows.append(
            [
                InlineKeyboardButton(
                    text=M._clip_label("", f"{preset.icon} {preset.name}"),
                    callback_data=f"preset:view:{preset.id}",
                )
            ]
        )
    pager = M._pager_row(current, total, "preset_page")
    if pager is not None:
        rows.append(pager)
    rows.append([InlineKeyboardButton(text="← Назад", callback_data="settings:open")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _preset_preview_parts(
    preset: AssistantPreset, assistant: AssistAIClient
) -> tuple[str, InlineKeyboardMarkup]:
    """Предпросмотр пресета: модели и полный промпт."""
    mode_label = _VOICE_MODE_LABELS.get(preset.voice_mode, preset.voice_mode)
    text_line = M._chosen_model_line(
        preset.text_model, assistant.config.default_text_model
    )
    vision_line = (
        M._chosen_model_line(preset.vision_model, assistant.config.default_text_model)
        if preset.vision_model
        else "—"
    )
    image_line = (
        M._chosen_model_line(preset.image_model, assistant.config.default_image_model)
        if preset.image_model
        else "—"
    )
    size_line = preset.image_size or "—"
    tts_line = (
        M._chosen_model_line(preset.tts_model, assistant.config.default_tts_model)
        if preset.tts_model
        else "—"
    )
    voice_line = preset.tts_voice or "—"
    header = (
        f"{preset.icon} {preset.name}\n"
        f"{preset.summary}\n\n"
        f"Текст: {text_line}\n"
        f"Vision: {vision_line}\n"
        f"Картинка: {image_line}\n"
        f"Размер: {size_line}\n"
        f"Голос: {mode_label}\n"
        f"TTS: {tts_line}\n"
        f"TTS голос: {voice_line}\n\n"
    )
    prompt = preset.system_prompt
    room = TELEGRAM_TEXT_LIMIT - len(header) - 1
    if room < 32:
        prompt = "…"
    elif len(prompt) > room:
        prompt = prompt[: room - 1] + "…"
    text = f"{header}{prompt}"
    markup = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="✅ Создать",
                    callback_data=f"preset:apply:{preset.id}",
                )
            ],
            [
                InlineKeyboardButton(
                    text="✏️ Создать и настроить",
                    callback_data=f"preset:edit:{preset.id}",
                )
            ],
            [
                InlineKeyboardButton(
                    text="← К каталогу",
                    callback_data="preset:list",
                )
            ],
        ]
    )
    return text, markup
