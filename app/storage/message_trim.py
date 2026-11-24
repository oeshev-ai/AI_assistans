"""Обрезка ленты сообщений OpenAI до лимита контекста."""

from __future__ import annotations

from typing import Any


def _drop_leading_tools(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Убирает ``role: tool`` в начале: без своего ``tool_calls`` это сироты."""
    start = next(
        (index for index, item in enumerate(messages) if item.get("role") != "tool"),
        None,
    )
    if start is None:
        return []
    return messages[start:]


def trim_messages(
    messages: list[dict[str, Any]], max_length: int
) -> list[dict[str, Any]]:
    """Оставляет не больше ``max_length`` сообщений, самые новые.

    Системный промпт в начале списка не вытесняется. После среза история
    выравнивается на ближайший ``user``: иначе посередине tool-раунда
    остаются сироты ``role: tool`` без ``tool_calls``, и ProxyAPI отвечает 400.
    Если в срезе нет ``user``, сохраняется весь ход, к которому относится
    хвост, даже длиннее лимита. Иначе длинный tool-раунд оставляет только
    system и стирает контекст темы.
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
    trimmed = body[-max_body:] if len(body) > max_body else body
    start = next(
        (index for index, item in enumerate(trimmed) if item.get("role") == "user"),
        None,
    )
    if start is not None:
        trimmed = trimmed[start:]
    else:
        cut = len(body) - len(trimmed)
        prior = next(
            (
                index
                for index in range(cut - 1, -1, -1)
                if body[index].get("role") == "user"
            ),
            None,
        )
        trimmed = body[prior:] if prior is not None else _drop_leading_tools(trimmed)
    if system is None:
        return trimmed
    return [system, *trimmed]
