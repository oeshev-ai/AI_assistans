"""Списки моделей ProxyAPI для кнопок бота.

Идентификаторы взяты из каталога https://proxyapi.ru/models/browse.
Каталог не запрашивается при каждом нажатии. Пустой выбор в профиле
означает модель по умолчанию из ``AssistAIConfig``.
"""

from __future__ import annotations

AVAILABLE_TEXT_MODELS: tuple[str, ...] = (
    "qwen/qwen3.8-flash",
    "deepseek/deepseek-v4.1-flash",
    "openai/gpt-6-luna",
    "anthropic/claude-haiku-5-5",
    "z-ai/glm-5.3-flash",
    "google/gemini-3.1-flash-lite",
    "google/gemini-3.5-flash-lite",
    "qwen/qwen3.8-max-0902",
    "google/gemini-3.8-flash",
    "anthropic/claude-sonnet-5-5",
    "openai/gpt-6.1-sol",
    "openai/gpt-6-astra",
    "google/gemini-4-argon",
)

# Картинки: OpenAI и FLUX — как раньше; Qwen/Seedream/Recraft — свои поля размера.
AVAILABLE_IMAGE_MODELS: tuple[str, ...] = (
    "openai/gpt-image-1-mini",
    "openai/gpt-image-1.5",
    "openai/gpt-image-2",
    "black-forest-labs/flux-3-image-klein",
    "black-forest-labs/flux-3-image",
    "qwen/qwen-image-2.1-turbo",
    "qwen/qwen-image-2.1-pro",
    "bytedance-seed/seedream-5-0-flash",
    "recraft/recraft-v4.1",
)
