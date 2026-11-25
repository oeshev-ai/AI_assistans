"""Длина текста так, как её считает Telegram Bot API.

Лимит сообщения — 4096 UTF-16 code units. ``len(str)`` считает code points,
поэтому эмодзи и прочие символы вне BMP (по 2 units) проскакивают потолок
и Bot API отвечает «message is too long».
"""

from __future__ import annotations

# Потолок sendMessage / editMessageText.
TELEGRAM_TEXT_LIMIT = 4096
# Запас ниже потолка: отказ доставки откатывает уже оплаченный ход.
TELEGRAM_TEXT_SAFE = 4000


def telegram_text_units(text: str) -> int:
    """Длина в UTF-16 code units."""
    return len(text.encode("utf-16-le", errors="surrogatepass")) // 2


def clip_telegram_text(text: str, limit: int = TELEGRAM_TEXT_SAFE) -> str:
    """Префикс не длиннее ``limit`` UTF-16 code units.

    Граница не режет суррогатную пару: если срез попадает на старший
    суррогат, он отбрасывается.
    """
    if limit <= 0 or not text:
        return ""
    encoded = text.encode("utf-16-le", errors="surrogatepass")
    if len(encoded) <= limit * 2:
        return text
    raw = encoded[: limit * 2]
    if len(raw) >= 2:
        unit = int.from_bytes(raw[-2:], "little")
        if 0xD800 <= unit <= 0xDBFF:
            raw = raw[:-2]
    return raw.decode("utf-16-le", errors="surrogatepass")


def split_telegram_text(text: str, limit: int = TELEGRAM_TEXT_SAFE) -> list[str]:
    """Режет текст на куски не длиннее ``limit`` UTF-16 code units."""
    if limit <= 0 or not text:
        return []
    parts: list[str] = []
    rest = text
    while rest:
        chunk = clip_telegram_text(rest, limit)
        if not chunk:
            break
        parts.append(chunk)
        rest = rest[len(chunk) :]
    return parts
