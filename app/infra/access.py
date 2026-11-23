"""Telegram user allowlist for API cost protection."""

from __future__ import annotations

import structlog

logger = structlog.get_logger("access")


def is_production() -> bool:
    """True when ``ENVIRONMENT=production`` (compose sets this on bot/worker)."""
    from app.infra.settings import get_settings

    return get_settings().is_production


def require_redis_password() -> None:
    """Завершает процесс, если production запущен без ``REDIS_PASSWORD``.

    Пустой пароль поднимает Redis без ``requirepass``. Локально
    ``ENVIRONMENT`` не production, и пустое значение остаётся допустимым.
    """
    from app.infra.settings import get_settings

    settings = get_settings()
    if not settings.is_production or settings.redis_password:
        return
    logger.error(
        "redis_password_required_in_production",
        hint="Set REDIS_PASSWORD in .env. Production Redis must use requirepass.",
    )
    raise SystemExit(1)


def parse_allowed_user_ids(raw: str | None = None) -> frozenset[int]:
    """Parse ``TELEGRAM_ALLOWED_USER_IDS`` (comma/space separated ints).

    Empty / unset → empty frozenset (allow-all mode for local/dev).
    Invalid tokens are skipped with a warning.
    """
    if raw is None:
        from app.infra.settings import get_settings

        raw = get_settings().telegram_allowed_user_ids
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
