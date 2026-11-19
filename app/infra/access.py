"""Telegram user allowlist for API cost protection."""

from __future__ import annotations

import os

import structlog

logger = structlog.get_logger("access")


def is_production() -> bool:
    """True when ``ENVIRONMENT=production`` (compose sets this on bot/worker)."""
    return os.getenv("ENVIRONMENT", "development").strip().lower() == "production"


def parse_allowed_user_ids(raw: str | None = None) -> frozenset[int]:
    """Parse ``TELEGRAM_ALLOWED_USER_IDS`` (comma/space separated ints).

    Empty / unset → empty frozenset (allow-all mode for local/dev).
    Invalid tokens are skipped with a warning.
    """
    if raw is None:
        raw = os.getenv("TELEGRAM_ALLOWED_USER_IDS", "")
    allowed: set[int] = set()
    skipped: list[str] = []
    for part in raw.replace(";", ",").replace(" ", ",").split(","):
        token = part.strip()
        if not token:
            continue
        try:
            allowed.add(int(token))
        except ValueError:
            skipped.append(token)
    if skipped:
        logger.warning("allowlist_invalid_tokens", tokens=skipped)
    return frozenset(allowed)


def is_user_allowed(user_id: int | None, allowed: frozenset[int]) -> bool:
    """Return True if ``allowed`` is empty (open) or ``user_id`` is listed."""
    if user_id is None:
        return not bool(allowed)
    if not allowed:
        return True
    return user_id in allowed
