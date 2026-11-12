"""Structured JSON logging via structlog."""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: int = logging.INFO) -> None:
    """Configure structlog + stdlib so all logs emit JSON to stdout."""
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.format_exc_info,
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("aiohttp.access").setLevel(logging.WARNING)


def bind_request_context(
    *,
    user_id: int | None = None,
    chat_id: int | None = None,
    assistant_id: str | None = None,
    topic_id: str | None = None,
) -> None:
    """Attach request-scoped fields to subsequent log records."""
    payload: dict[str, int | str] = {}
    if user_id is not None:
        payload["user_id"] = user_id
    if chat_id is not None:
        payload["chat_id"] = chat_id
    if assistant_id is not None:
        payload["assistant_id"] = assistant_id
    if topic_id is not None:
        payload["topic_id"] = topic_id
    if payload:
        structlog.contextvars.bind_contextvars(**payload)


def clear_request_context() -> None:
    """Drop request-scoped context after a handler finishes."""
    structlog.contextvars.clear_contextvars()
