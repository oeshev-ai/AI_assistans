"""Bot-level middlewares (access control)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import structlog
from aiogram import BaseMiddleware
from aiogram.types import CallbackQuery, Message, TelegramObject

from app.infra.access import is_user_allowed

logger = structlog.get_logger("bot.middleware")

_DENIED_TEXT = "Бот доступен только авторизованным пользователям."


class AllowlistMiddleware(BaseMiddleware):
    """Drop updates from users not in ``TELEGRAM_ALLOWED_USER_IDS``.

    Empty allowlist → everyone is allowed (local/dev default).
    """

    def __init__(self, allowed_ids: frozenset[int]) -> None:
        self._allowed = allowed_ids

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        if not self._allowed:
            return await handler(event, data)

        user = data.get("event_from_user")
        user_id = getattr(user, "id", None) if user is not None else None
        if user_id is None:
            if isinstance(event, Message) and event.from_user is not None:
                user_id = event.from_user.id
            elif isinstance(event, CallbackQuery) and event.from_user is not None:
                user_id = event.from_user.id

        if is_user_allowed(user_id, self._allowed):
            return await handler(event, data)

        logger.warning("allowlist_denied", user_id=user_id)
        if isinstance(event, Message):
            try:
                await event.answer(_DENIED_TEXT)
            except Exception:
                logger.exception("allowlist_deny_reply_failed")
        elif isinstance(event, CallbackQuery):
            try:
                await event.answer(_DENIED_TEXT, show_alert=True)
            except Exception:
                logger.exception("allowlist_deny_callback_failed")
        return None
