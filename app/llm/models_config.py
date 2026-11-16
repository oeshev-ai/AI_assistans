"""Списки моделей ProxyAPI для кнопок бота.

Идентификаторы взяты из каталога https://proxyapi.ru/models/browse.
Каталог не запрашивается при каждом нажатии. Пустой выбор в профиле
означает модель по умолчанию из ``AssistAIConfig``.
"""

from __future__ import annotations

import os

AVAILABLE_TEXT_MODELS: tuple[str, ...] = (
    "deepseek/deepseek-v4.1-flash",
    "google/gemini-3.8-flash",
    "google/gemini-4-argon",
    "anthropic/claude-sonnet-5-5",
    "openai/gpt-6-luna",
    "openai/gpt-6.1-terra",
    "openai/gpt-6-sol",
    "qwen/qwen3.8-max-0902",
)

# Картинки: GPT Image 2.6 — основная; FLUX, Qwen и Recraft — свои поля размера.
AVAILABLE_IMAGE_MODELS: tuple[str, ...] = (
    "openai/gpt-image-2.6-flare",
    "black-forest-labs/flux-3-image-klein",
    "black-forest-labs/flux-3-image",
    "qwen/qwen-image-2.1-pro",
    "recraft/recraft-v4.1",
)

# Озвучка: SpeechKit + модели ProxyAPI /audio/speech.
SPEECHKIT_TTS_MODEL = "yandex/speechkit"
AVAILABLE_TTS_MODELS: tuple[str, ...] = (
    SPEECHKIT_TTS_MODEL,
    "openai/gpt-4o-mini-tts",
    "elevenlabs/eleven-multilingual-v2",
)

# Голоса ProxyAPI OpenAI TTS (alloy и др.).
OPENAI_TTS_VOICES: tuple[str, ...] = ("alloy", "verse", "nova", "shimmer")

# Голоса ProxyAPI ElevenLabs multilingual v2 (см. карточку модели).
ELEVENLABS_TTS_VOICES: tuple[str, ...] = (
    "george",
    "sarah",
    "adam",
    "alice",
    "bella",
    "brian",
    "charlie",
    "daniel",
    "jessica",
    "laura",
    "liam",
    "lily",
    "matilda",
    "river",
    "will",
)

MODEL_HINTS: dict[str, str] = {
    # Текст
    "deepseek/deepseek-v4.1-flash": "Дёшево: логика, код, математика, чёткая структура",
    "openai/gpt-6-luna": "Универсальный ассистент, аккуратно вызывает инструменты",
    "qwen/qwen3.8-max-0902": "Сложные тексты, аналитика, переводы со стилем",
    "google/gemini-3.8-flash": "Баланс: длинный контекст, документы, обучение",
    "anthropic/claude-sonnet-5-5": "Код, ревью, рефакторинг, точные инструкции",
    "openai/gpt-6-sol": "Флагман: сложные рассуждения и аналитика",
    "openai/gpt-6.1-terra": "Флагман: тексты, креатив, агентные задачи",
    "google/gemini-4-argon": "Старший Gemini: фото, PDF, очень длинный контекст",
    # Картинки
    "openai/gpt-image-2.6-flare": "Универсал: фото, правки, надписи, несколько фото",
    "black-forest-labs/flux-3-image-klein": "Быстрый FLUX: наброски, варианты, итерации",
    "black-forest-labs/flux-3-image": "Арт и фотореализм: постеры, иллюстрации",
    "qwen/qwen-image-2.1-pro": "Типографика и текст: афиши, обложки, карточки",
    "recraft/recraft-v4.1": "Дизайн-графика: логотипы, иконки, схемы, вектор",
}


def is_elevenlabs_tts_model(model: str | None) -> bool:
    """True для моделей ElevenLabs через ProxyAPI."""
    return (model or "").strip().casefold().startswith("elevenlabs/")


LEGACY_IMAGE_MODEL_ALIASES: dict[str, str] = {
    "openai/gpt-image-1-mini": "openai/gpt-image-2.6-flare",
    "openai/gpt-image-1.5": "openai/gpt-image-2.6-flare",
    "openai/gpt-image-2": "openai/gpt-image-2.6-flare",
    "openai/gpt-image-2.5-flare": "openai/gpt-image-2.6-flare",
    "qwen/qwen-image-2.1-turbo": "qwen/qwen-image-2.1-pro",
    "bytedance-seed/seedream-5-0-flash": "black-forest-labs/flux-3-image-klein",
}

LEGACY_TEXT_MODEL_ALIASES: dict[str, str] = {
    "qwen/qwen3.8-flash": "deepseek/deepseek-v4.1-flash",
    "z-ai/glm-5.3-flash": "deepseek/deepseek-v4.1-flash",
    "google/gemini-3.1-flash-lite": "google/gemini-3.8-flash",
    "google/gemini-3.5-flash-lite": "google/gemini-3.8-flash",
    "anthropic/claude-haiku-5-5": "google/gemini-3.8-flash",
}


def _aliases_enabled(env_name: str) -> bool:
    return os.getenv(env_name, "1").strip().casefold() not in {"0", "false", "no"}


def resolve_image_model(model_id: str | None) -> str | None:
    if not model_id or not _aliases_enabled("IMAGE_LEGACY_ALIASES"):
        return model_id
    return LEGACY_IMAGE_MODEL_ALIASES.get(model_id.strip().casefold(), model_id)


def resolve_text_model(model_id: str | None) -> str | None:
    if not model_id or not _aliases_enabled("TEXT_LEGACY_ALIASES"):
        return model_id
    return LEGACY_TEXT_MODEL_ALIASES.get(model_id.strip().casefold(), model_id)


def model_hint(model_id: str | None) -> str:
    """Подсказка модели. Неизвестная или пустая модель → пустая строка."""
    if not model_id:
        return ""
    return MODEL_HINTS.get(model_id, "")


def validate_hints() -> None:
    """Проверяет, что у каждой text/image-модели есть подсказка 1..60 символов.

    Лишних ключей в ``MODEL_HINTS`` быть не должно. Ошибка — ``RuntimeError``.
    """
    catalog = set(AVAILABLE_TEXT_MODELS) | set(AVAILABLE_IMAGE_MODELS)
    for model_id in catalog:
        hint = MODEL_HINTS.get(model_id)
        if hint is None:
            raise RuntimeError(f"MODEL_HINTS: нет подсказки для {model_id}")
        length = len(hint)
        if length < 1 or length > 60:
            raise RuntimeError(
                f"MODEL_HINTS: длина подсказки для {model_id} = {length}, нужно 1..60"
            )
    extra = set(MODEL_HINTS) - catalog
    if extra:
        raise RuntimeError(f"MODEL_HINTS: лишние ключи вне каталога: {sorted(extra)}")


validate_hints()
