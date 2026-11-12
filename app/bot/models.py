"""Model/size helpers and hint formatting."""

from __future__ import annotations

from app.bot.constants import *


def _text_model_of(
    profile: AssistantProfile,
    assistant: AssistAIClient,
    *,
    chat_id: int | None = None,
) -> str:
    """Модель диалога: выбор профиля или дефолт клиента.

    Непустой id вне каталога не пишется в Redis — только мягкий fallback.
    """
    selected = profile.text_model
    if not selected:
        return assistant.config.default_text_model
    if selected in AVAILABLE_TEXT_MODELS:
        return selected
    logger.warning(
        "stale_model_fallback",
        chat_id=chat_id,
        model=selected,
        kind="text",
    )
    return assistant.config.default_text_model


def _image_model_of(
    profile: AssistantProfile,
    assistant: AssistAIClient,
    *,
    chat_id: int | None = None,
) -> str:
    """Модель картинки: выбор профиля или дефолт клиента.

    Непустой id вне каталога не пишется в Redis — только мягкий fallback.
    """
    selected = profile.image_model
    if not selected:
        return assistant.config.default_image_model
    if selected in AVAILABLE_IMAGE_MODELS:
        return selected
    logger.warning(
        "stale_model_fallback",
        chat_id=chat_id,
        model=selected,
        kind="image",
    )
    return assistant.config.default_image_model


def _menu_button(title: str, name: str) -> str:
    """Подпись нижней кнопки с именем. Telegram режет текст длиннее 64 символов."""
    prefix = f"{title} ["
    suffix = "]"
    room = 64 - len(prefix) - len(suffix)
    if room < 1:
        return title[:64]
    shown = name if len(name) <= room else f"{name[: room - 3]}..."
    return f"{prefix}{shown}{suffix}"


def _topic_display_name(name: str) -> str:
    """Старое имя General показываем как «Общая»."""
    return "Общая" if name == "General" else name


def _reply_menu(assistant_name: str, topic_name: str) -> ReplyKeyboardMarkup:
    """Две кнопки в два ряда: персоны и темы. Настройки — в меню команд."""
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text=_menu_button(_MENU_ASSISTANTS, assistant_name))],
            [KeyboardButton(text=_menu_button(_MENU_TOPICS, topic_name))],
        ],
        resize_keyboard=True,
        is_persistent=True,
    )


def _size_choices(model_id: str) -> tuple[str, ...]:
    """Кнопки размера зависят от семейства image-модели."""
    family = image_model_family(model_id)
    if family in {"flux", "seedream"}:
        return _TIER_IMAGE_SIZES
    if family in {"qwen", "recraft"}:
        return _ASPECT_IMAGE_SIZES
    return _OPENAI_IMAGE_SIZES


def _default_image_size(model_id: str) -> str:
    family = image_model_family(model_id)
    if family in {"flux", "seedream"}:
        return "1K"
    if family in {"qwen", "recraft"}:
        return "1:1"
    return "1024x1024"


def _display_image_size(profile: AssistantProfile, assistant: AssistAIClient) -> str:
    """Размер для экрана настроек. Пустой или чужой семье размер показывает дефолт."""
    model_id = _image_model_of(profile, assistant)
    if profile.image_size in _size_choices(model_id):
        return profile.image_size or _default_image_size(model_id)
    return _default_image_size(model_id)


def _stored_image_size(profile: AssistantProfile) -> str:
    """Размер для генерации. Пустое поле персоны остаётся 1024x1024."""
    if isinstance(profile.image_size, str) and profile.image_size.strip():
        return profile.image_size.strip()
    return "1024x1024"


def _vision_model_of(profile: AssistantProfile, assistant: AssistAIClient) -> str:
    """Модель анализа картинок: vision → text → дефолт клиента."""
    return (
        profile.vision_model
        or profile.text_model
        or assistant.config.default_text_model
    )


def _should_send_voice_reply(profile: AssistantProfile, input_kind: InputKind) -> bool:
    """Нужен ли голосовой ответ поверх текста."""
    if profile.voice_mode == "off":
        return False
    if profile.voice_mode == "always":
        return True
    return profile.voice_mode == "auto" and input_kind == "voice"


def _clip_tts_text(text: str) -> str:
    """Обрезает длинный ответ для TTS по последнему пробелу."""
    cleaned = text.strip()
    if len(cleaned) <= TTS_MAX_CHARS:
        return cleaned
    cut = cleaned[:TTS_MAX_CHARS]
    space = cut.rfind(" ")
    if space > TTS_MAX_CHARS // 2:
        cut = cut[:space]
    logger.warning(
        "TTS обрезан: было %s символов, озвучено %s",
        len(cleaned),
        len(cut),
    )
    return cut.rstrip()


def _tts_model_of(profile: AssistantProfile, assistant: AssistAIClient) -> str:
    """Эффективная TTS-модель персоны (пустой выбор = дефолт конфига)."""
    return profile.tts_model or assistant.config.default_tts_model


def _tts_voices_for(
    assistant: AssistAIClient, *, model_id: str | None = None
) -> tuple[str, ...]:
    """Список голосов для UI в зависимости от TTS-модели."""
    effective = model_id or assistant.config.default_tts_model
    if is_speechkit_tts_model(effective):
        return SPEECHKIT_VOICES
    if is_elevenlabs_tts_model(effective):
        return ELEVENLABS_TTS_VOICES
    return OPENAI_TTS_VOICES


def _resolve_profile_tts_voice(
    profile: AssistantProfile, assistant: AssistAIClient
) -> str:
    """Голос профиля с миграцией при смене семейства TTS-модели."""
    model_id = _tts_model_of(profile, assistant)
    raw = profile.tts_voice or assistant.config.default_tts_voice
    if is_speechkit_tts_model(model_id):
        default = (
            assistant.config.default_tts_voice
            if is_speechkit_tts_model(assistant.config.default_tts_model)
            else "marina"
        )
        return resolve_speechkit_voice(raw, default)
    voices = _tts_voices_for(assistant, model_id=model_id)
    if raw in voices:
        return raw
    if assistant.config.default_tts_voice in voices:
        return assistant.config.default_tts_voice
    return voices[0]


def _compatible_tts_voice(
    profile: AssistantProfile,
    assistant: AssistAIClient,
    *,
    new_model: str | None,
) -> str | None:
    """Голос после смены TTS-модели; сбрасывает несовместимый."""
    effective = new_model or assistant.config.default_tts_model
    voices = _tts_voices_for(assistant, model_id=effective)
    current = profile.tts_voice
    if current in voices:
        return current
    return None


def _model_keyboard(
    models: tuple[str, ...],
    prefix: str,
) -> InlineKeyboardMarkup:
    """Кнопки моделей. В callback кладётся индекс, не имя."""
    rows = [
        [InlineKeyboardButton(text=model, callback_data=f"{prefix}:{index}")]
        for index, model in enumerate(models)
    ]
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _chosen_model_line(selected: str | None, fallback: str) -> str:
    if selected is None:
        return f"{fallback} (по умолчанию)"
    return selected


def _hints_block(models: tuple[str, ...], current: str | None = None) -> str:
    """Строки «• {model} — {hint}». У текущей модели префикс ✅ вместо •."""
    lines: list[str] = []
    for model_id in models:
        prefix = "✅" if current is not None and model_id == current else "•"
        hint = model_hint(model_id)
        if hint:
            lines.append(f"{prefix} {model_id} — {hint}")
        else:
            lines.append(f"{prefix} {model_id}")
    return "\n".join(lines)


def _fit_screen(text: str) -> str:
    """Обрезает текст до TELEGRAM_TEXT_LIMIT с «…» на конце."""
    if len(text) <= TELEGRAM_TEXT_LIMIT:
        return text
    if TELEGRAM_TEXT_LIMIT <= 1:
        return "…"
    return text[: TELEGRAM_TEXT_LIMIT - 1] + "…"


def _clip_label(prefix: str, name: str, *, suffix: str = "") -> str:
    """Подпись кнопки не длиннее 64 символов. ``suffix`` (например `` 🖼``) не обрезается."""
    label = f"{prefix}{name}{suffix}"
    if len(label) <= 64:
        return label
    keep = 64 - len(prefix) - len(suffix) - 3
    if keep < 1:
        return label[:64]
    return f"{prefix}{name[:keep]}...{suffix}"


def _optional_model(value: Any) -> str | None:
    """Пустая строка и не-строка означают модель по умолчанию."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _page_bounds(count: int, page: int) -> tuple[int, int]:
    """Номер страницы в допустимых пределах и общее число страниц."""
    total = max(1, (count + PAGE_SIZE - 1) // PAGE_SIZE)
    current = max(0, min(page, total - 1))
    return current, total


def _pager_row(page: int, total: int, prefix: str) -> list[InlineKeyboardButton] | None:
    """Строка ◀ / ▶. На одной странице её нет."""
    if total <= 1:
        return None
    row: list[InlineKeyboardButton] = []
    if page > 0:
        row.append(InlineKeyboardButton(text="◀", callback_data=f"{prefix}:{page - 1}"))
    if page < total - 1:
        row.append(InlineKeyboardButton(text="▶", callback_data=f"{prefix}:{page + 1}"))
    return row or None


def _confirm_row(
    yes_text: str, yes_data: str, cancel_data: str
) -> InlineKeyboardMarkup:
    """Пара кнопок: подтвердить действие или вернуться к списку."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text=yes_text, callback_data=yes_data),
                InlineKeyboardButton(text="← Отмена", callback_data=cancel_data),
            ]
        ]
    )
