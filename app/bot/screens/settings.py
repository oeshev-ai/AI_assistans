"""Screen builders."""

from __future__ import annotations

from collections.abc import Callable

from app.bot import models as M
from app.bot.constants import (
    _VOICE_MODE_LABELS,
    AVAILABLE_IMAGE_MODELS,
    AVAILABLE_TEXT_MODELS,
    AVAILABLE_TTS_MODELS,
    DEFAULT_ASSISTANT_ID,
    AssistAIClient,
    AssistantProfile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    is_speechkit_tts_model,
    model_hint,
    persona_label,
    resolve_speechkit_role,
    speechkit_roles_for,
)
from app.llm.video_settings import (
    VIDEO_ASPECT_RATIOS,
    VIDEO_DURATIONS,
    VIDEO_RESOLUTIONS,
    video_settings_from,
    video_settings_summary,
)


def _choice_rows(
    items: tuple[object, ...],
    selected: object,
    *,
    prefix: str,
    label_of: Callable[[object], str],
    per_row: int,
) -> list[list[InlineKeyboardButton]]:
    """Ряды кнопок выбора. Текущее значение помечено галочкой."""
    rows: list[list[InlineKeyboardButton]] = []
    row: list[InlineKeyboardButton] = []
    for item in items:
        mark = "✅ " if item == selected else ""
        row.append(
            InlineKeyboardButton(
                text=f"{mark}{label_of(item)}",
                callback_data=f"{prefix}{item}",
            )
        )
        if len(row) >= per_row:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    return rows


def _video_picker_parts(
    profile: AssistantProfile,
) -> tuple[str, InlineKeyboardMarkup]:
    """Экран формата, длины, разрешения и звука ролика активной персоны."""
    settings = video_settings_from(profile)
    sound = "вкл" if settings.generate_audio else "выкл"
    rows = _choice_rows(
        VIDEO_ASPECT_RATIOS,
        settings.aspect_ratio,
        prefix="video:aspect:",
        label_of=str,
        per_row=3,
    )
    rows.extend(
        _choice_rows(
            VIDEO_DURATIONS,
            settings.duration,
            prefix="video:dur:",
            label_of=lambda item: f"{item}с",
            per_row=6,
        )
    )
    rows.extend(
        _choice_rows(
            VIDEO_RESOLUTIONS,
            settings.resolution,
            prefix="video:res:",
            label_of=str,
            per_row=2,
        )
    )
    audio_row: list[InlineKeyboardButton] = []
    for enabled, label in ((False, "Без звука"), (True, "Со звуком")):
        mark = "✅ " if settings.generate_audio is enabled else ""
        flag = "1" if enabled else "0"
        audio_row.append(
            InlineKeyboardButton(
                text=f"{mark}{label}",
                callback_data=f"video:audio:{flag}",
            )
        )
    rows.append(audio_row)
    rows.append(
        [
            InlineKeyboardButton(
                text="Готово", callback_data="picker:done", style="success"
            )
        ]
    )
    text = (
        f"Видео — «{profile.name}»\n"
        f"Формат: {settings.aspect_ratio}\n"
        f"Длительность: {settings.duration} с\n"
        f"Разрешение: {settings.resolution}\n"
        f"Звук: {sound}\n\n"
        "Выберите формат, длительность, разрешение или звук."
    )
    return text, InlineKeyboardMarkup(inline_keyboard=rows)


def _model_picker_parts(
    profile: AssistantProfile, assistant: AssistAIClient, kind: str
) -> tuple[str, InlineKeyboardMarkup]:
    """Текст и кнопки выбора модели активной персоны."""
    rows: list[list[InlineKeyboardButton]] = []
    if kind == "video":
        return _video_picker_parts(profile)
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
        background = M._image_background_of(profile, assistant)
        background_line = ""
        choose = "Выберите модель, размер или получение файлом:"
        if background is not None:
            bg_row: list[InlineKeyboardButton] = []
            for value, label in M._BACKGROUND_LABELS.items():
                mark = "✅ " if value == background else ""
                bg_row.append(
                    InlineKeyboardButton(
                        text=f"{mark}{label}",
                        callback_data=f"set_bg:{value}",
                    )
                )
            rows.append(bg_row)
            background_line = f"Фон: {M._background_label(background)}\n"
            choose = "Выберите модель, размер, фон или получение файлом:"
        file_on = M._image_as_file(profile, image_model)
        file_row: list[InlineKeyboardButton] = []
        for enabled, label in ((False, "Фото"), (True, "Файлом")):
            mark = "✅ " if file_on is enabled else ""
            flag = "1" if enabled else "0"
            file_row.append(
                InlineKeyboardButton(
                    text=f"{mark}{label}",
                    callback_data=f"set_file:{flag}",
                )
            )
        rows.append(file_row)
        file_line = f"Получать результат файлом: {M._image_file_label(file_on)}\n"
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
            f"Размер: {shown_size}\n"
            f"{background_line}"
            f"{file_line}\n"
            f"{choose}\n\n"
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
    keyboard = M._model_keyboard(AVAILABLE_TEXT_MODELS, "set_text", done="picker:done")
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
        ]
    )
    background = M._image_background_of(profile, assistant)
    if background is not None:
        lines.append(f"Фон картинки: {M._background_label(background)}")
    lines.append(
        "Получать результат файлом: "
        f"{M._image_file_label(M._image_as_file(profile, image_model))}"
    )
    lines.extend(
        [
            f"Видео: {video_settings_summary(video_settings_from(profile))}",
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
                text="Видео",
                callback_data="video_menu",
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
