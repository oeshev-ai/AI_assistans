"""Screen builders."""

from __future__ import annotations

from app.bot import models as M
from app.bot.constants import *


def _model_picker_parts(
    profile: AssistantProfile, assistant: AssistAIClient, kind: str
) -> tuple[str, InlineKeyboardMarkup]:
    """Текст и кнопки выбора модели активной персоны."""
    rows: list[list[InlineKeyboardButton]] = []
    if kind == "image":
        current = M._chosen_model_line(
            profile.image_model, assistant.config.default_image_model
        )
        shown_size = M._display_image_size(profile, assistant)
        image_model = M._image_model_of(profile, assistant)
        for index, model in enumerate(AVAILABLE_IMAGE_MODELS):
            mark = "✅ " if image_model == model else ""
            rows.append(
                [
                    InlineKeyboardButton(
                        text=M._clip_label(mark, model),
                        callback_data=f"set_image:{index}",
                    )
                ]
            )
        size_row: list[InlineKeyboardButton] = []
        for item in M._size_choices(image_model):
            mark = "✅ " if item == shown_size else ""
            size_row.append(
                InlineKeyboardButton(
                    text=f"{mark}{item}", callback_data=f"set_size:{item}"
                )
            )
        rows.append(size_row)
        rows.append(
            [
                InlineKeyboardButton(
                    text="Готово", callback_data="picker:done", style="success"
                )
            ]
        )
        text = (
            f"Модель картинок ассистента «{profile.name}»:\n"
            f"{current}\n"
            f"Размер: {shown_size}\n\n"
            "Выберите модель или размер:\n\n"
            f"{M._hints_block(AVAILABLE_IMAGE_MODELS, image_model)}"
        )
        return text, InlineKeyboardMarkup(inline_keyboard=rows)
    if kind == "tts":
        current = M._chosen_model_line(
            profile.tts_model, assistant.config.default_tts_model
        )
        tts_model = M._tts_model_of(profile, assistant)
        tts_voice = M._resolve_profile_tts_voice(profile, assistant)
        mode_label = _VOICE_MODE_LABELS.get(profile.voice_mode, profile.voice_mode)
        for index, model in enumerate(AVAILABLE_TTS_MODELS):
            mark = "✅ " if tts_model == model else ""
            rows.append(
                [
                    InlineKeyboardButton(
                        text=M._clip_label(mark, model),
                        callback_data=f"set_tts:{index}",
                    )
                ]
            )
        mode_row: list[InlineKeyboardButton] = []
        for mode in ("off", "auto", "always"):
            mark = "✅ " if profile.voice_mode == mode else ""
            mode_row.append(
                InlineKeyboardButton(
                    text=f"{mark}{_VOICE_MODE_LABELS[mode]}",
                    callback_data=f"voice_mode:{mode}",
                )
            )
        rows.append(mode_row)
        voices = M._tts_voices_for(assistant, model_id=tts_model)
        voice_row: list[InlineKeyboardButton] = []
        for index, name in enumerate(voices):
            mark = "✅ " if tts_voice == name else ""
            voice_row.append(
                InlineKeyboardButton(
                    text=f"{mark}{name}",
                    callback_data=f"voice_name:{index}",
                )
            )
            if len(voice_row) >= 4:
                rows.append(voice_row)
                voice_row = []
        if voice_row:
            rows.append(voice_row)
        extra = ""
        if is_speechkit_tts_model(tts_model):
            roles = speechkit_roles_for(tts_voice)
            requested = profile.tts_role or assistant.config.speechkit_role
            effective_role = resolve_speechkit_role(tts_voice, requested)
            if roles:
                role_row: list[InlineKeyboardButton] = []
                for name in roles:
                    mark = "✅ " if effective_role == name else ""
                    role_row.append(
                        InlineKeyboardButton(
                            text=f"{mark}{name}",
                            callback_data=f"voice_role:{name}",
                        )
                    )
                    if len(role_row) >= 4:
                        rows.append(role_row)
                        role_row = []
                if role_row:
                    rows.append(role_row)
                role_label = effective_role or "дефолт голоса"
            else:
                role_label = "не поддерживается"
            extra = f"Амплуа: {role_label}\nФормат: opus\n"
        elif is_elevenlabs_tts_model(tts_model):
            extra = "Формат: mp3\n"
        else:
            fmt = profile.tts_format or assistant.config.default_tts_format
            extra = f"Формат: {fmt}\n"
        rows.append(
            [
                InlineKeyboardButton(
                    text="Готово", callback_data="picker:done", style="success"
                )
            ]
        )
        text = (
            f"Голосовой ответ — «{profile.name}»\n"
            f"Модель: {current}\n"
            f"Режим: {mode_label}\n"
            f"Голос: {tts_voice}\n"
            f"{extra}\n"
            "Выберите модель, режим, голос или амплуа:"
        )
        return text, InlineKeyboardMarkup(inline_keyboard=rows)
    current = M._chosen_model_line(
        profile.text_model, assistant.config.default_text_model
    )
    text_model = M._text_model_of(profile, assistant)
    keyboard = M._model_keyboard(AVAILABLE_TEXT_MODELS, "set_text")
    text = (
        f"Текстовая модель ассистента «{profile.name}»:\n{current}\n\n"
        f"Выберите модель:\n\n{M._hints_block(AVAILABLE_TEXT_MODELS, text_model)}"
    )
    return text, keyboard


def _settings_hub_view(profile: AssistantProfile) -> tuple[str, InlineKeyboardMarkup]:
    """Корневой экран /settings: настройка, создание, каталог."""
    label = persona_label(profile.name, profile.icon)
    text = f"Настройки\nАктивная персона: ✅ {label}"
    rows: list[list[InlineKeyboardButton]] = [
        [
            InlineKeyboardButton(
                text="Настройка активной персоны",
                callback_data="settings:persona",
            )
        ],
        [
            InlineKeyboardButton(
                text="Создать новую персону",
                callback_data="asst:new",
            )
        ],
        [
            InlineKeyboardButton(
                text="📚 Каталог персон",
                callback_data="preset:list",
            )
        ],
        [
            InlineKeyboardButton(
                text="Готово", callback_data="settings:close", style="success"
            )
        ],
    ]
    return text, InlineKeyboardMarkup(inline_keyboard=rows)


def _settings_persona_view(
    profile: AssistantProfile, assistant: AssistAIClient
) -> tuple[str, InlineKeyboardMarkup]:
    """Параметры активной персоны: иконка, имя, промпт, модели, голос."""
    shown_size = M._display_image_size(profile, assistant)
    mode_label = _VOICE_MODE_LABELS.get(profile.voice_mode, profile.voice_mode)
    label = persona_label(profile.name, profile.icon)
    text_model = M._text_model_of(profile, assistant)
    image_model = M._image_model_of(profile, assistant)
    text_hint = model_hint(text_model)
    image_hint = model_hint(image_model)
    lines = [
        f"Настройка персоны: ✅ {label}",
        (
            "Текстовая модель: "
            f"{M._chosen_model_line(profile.text_model, assistant.config.default_text_model)}"
        ),
    ]
    if text_hint:
        lines.append(f"↳ {text_hint}")
    lines.append(
        "Модель изображений: "
        f"{M._chosen_model_line(profile.image_model, assistant.config.default_image_model)}"
    )
    if image_hint:
        lines.append(f"↳ {image_hint}")
    lines.extend(
        [
            f"Размер картинки: {shown_size}",
            (
                "TTS модель: "
                f"{M._chosen_model_line(profile.tts_model, assistant.config.default_tts_model)}"
            ),
            f"Голосовой ответ: {mode_label}",
        ]
    )
    text = "\n".join(lines)
    rows: list[list[InlineKeyboardButton]] = [
        [InlineKeyboardButton(text="Иконка", callback_data="asst_icon_menu")],
        [
            InlineKeyboardButton(
                text="Имя",
                callback_data=f"asst_rename:{profile.id}",
            )
        ],
        [
            InlineKeyboardButton(
                text="Промпт",
                callback_data=f"asst_edit:{profile.id}",
            )
        ],
        [
            InlineKeyboardButton(
                text="Текстовая модель",
                callback_data="set_text_menu",
            )
        ],
        [
            InlineKeyboardButton(
                text="Модель изображений",
                callback_data="set_image_menu",
            )
        ],
        [
            InlineKeyboardButton(
                text="Голосовой ответ",
                callback_data="voice_menu",
            )
        ],
    ]
    if profile.id != DEFAULT_ASSISTANT_ID:
        rows.append(
            [
                InlineKeyboardButton(
                    text="Удалить персону",
                    callback_data=f"asst:del_ask:{profile.id}",
                    style="danger",
                )
            ]
        )
    rows.append(
        [
            InlineKeyboardButton(
                text="Готово", callback_data="settings:open", style="success"
            )
        ]
    )
    return text, InlineKeyboardMarkup(inline_keyboard=rows)
