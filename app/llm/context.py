"""Оценка объёма сообщений. Длина base64 в счёт не входит."""

from __future__ import annotations

import math
from typing import Any

CONTEXT_SAFETY_MARGIN_PCT = 15
RESPONSE_RESERVE_TOKENS = 4096

_IMAGE_TYPES = frozenset({"image_url", "input_image", "image"})


def content_chars(content: Any) -> int:
    """Символы текста. Картинка даёт 0: base64 и URL в объём не входят."""
    if content is None:
        return 0
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for part in content:
            if isinstance(part, str):
                total += len(part)
                continue
            if not isinstance(part, dict) or _is_image_part(part):
                if not isinstance(part, dict):
                    total += len(str(part))
                continue
            text = part.get("text")
            if isinstance(text, str):
                total += len(text)
            elif text is not None:
                total += len(str(text))
        return total
    return len(str(content))


def estimate_tokens(messages: list[dict[str, Any]], *, chars_per_token: float) -> int:
    """Грубая оценка входа: текст и аргументы инструментов, плюс фикс на картинку."""
    if chars_per_token <= 0:
        raise ValueError("chars_per_token должен быть больше 0")
    from app.infra.settings import get_settings

    image_tokens = int(get_settings().image_token_estimate)
    chars = 0
    images = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        chars += content_chars(content)
        chars += _tool_argument_chars(message)
        images += _image_count(content)
    text_tokens = math.ceil(chars / chars_per_token) if chars else 0
    return text_tokens + images * image_tokens


def _is_image_part(part: dict[str, Any]) -> bool:
    kind = str(part.get("type") or "")
    if kind in _IMAGE_TYPES:
        return True
    return "image_url" in part or "input_image" in part


def _image_count(content: Any) -> int:
    if not isinstance(content, list):
        return 0
    return sum(1 for part in content if isinstance(part, dict) and _is_image_part(part))


def _tool_argument_chars(message: dict[str, Any]) -> int:
    total = 0
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        if isinstance(function, dict):
            total += len(str(function.get("arguments") or ""))
    return total
