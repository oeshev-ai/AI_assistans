"""Safe environment variable parsers with defaults on bad input."""

from __future__ import annotations

import os

import structlog

logger = structlog.get_logger("envparse")


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(str(raw).strip())
    except ValueError:
        logger.warning("env_int_invalid", name=name, value=raw, default=default)
        return default


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(str(raw).strip())
    except ValueError:
        logger.warning("env_float_invalid", name=name, value=raw, default=default)
        return default


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
