"""Safe parsers for values already read by ``app.infra.settings``."""

from __future__ import annotations

import structlog

logger = structlog.get_logger("envparse")


def env_int(name: str, default: int) -> int:
    """Пустая переменная оставляет дефолт. Нецелое значение — ошибка старта."""
    from app.infra.settings import getenv

    raw = getenv(name)
    if raw is None or not str(raw).strip():
        return default
    text = str(raw).strip()
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"{name}={text!r} не целое число") from exc


def env_float(name: str, default: float) -> float:
    """Пустая переменная оставляет дефолт. Нечисло, включая «1,5», — ошибка."""
    from app.infra.settings import getenv

    raw = getenv(name)
    if raw is None or not str(raw).strip():
        return default
    text = str(raw).strip()
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"{name}={text!r} не число") from exc


def ensure_https_url(url: str, *, name: str, default: str) -> str:
    """Return ``url`` or ``default``; in production reject non-HTTPS endpoints."""
    cleaned = (url or "").strip() or default
    if cleaned.lower().startswith("https://"):
        return cleaned
    from app.infra.access import is_production

    if is_production():
        logger.error("https_url_required_in_production", name=name, value=cleaned)
        raise SystemExit(1)
    logger.warning(
        "insecure_url_allowed_outside_production",
        name=name,
        value=cleaned,
    )
    return cleaned
