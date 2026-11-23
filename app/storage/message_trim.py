"""Обрезка ленты сообщений OpenAI до лимита контекста."""

from __future__ import annotations

from typing import Any


def trim_messages(
    messages: list[dict[str, Any]], max_length: int
) -> list[dict[str, Any]]:
    """Оставляет не больше ``max_length`` сообщений, самые новые.

    Системный промпт в начале списка не вытесняется. После среза история
    выравнивается на ближайший ``user``: иначе посередине tool-раунда
    остаются сироты ``role: tool`` без ``tool_calls``, и ProxyAPI отвечает 400.
    """
    if max_length < 1:
        raise ValueError("max_length должен быть >= 1.")
    if len(messages) <= max_length:
        return [dict(item) for item in messages]

    has_system = bool(messages) and messages[0].get("role") == "system"
    system = dict(messages[0]) if has_system else None
    body = [dict(item) for item in (messages[1:] if has_system else messages)]
    max_body = max_length - (1 if has_system else 0)
    if max_body < 1:
        return [system] if system is not None else []
    if len(body) > max_body:
        body = body[-max_body:]
    start = next(
        (index for index, item in enumerate(body) if item.get("role") == "user"),
        None,
    )
    body = body[start:] if start is not None else []
    if system is None:
        return body
    return [system, *body]
