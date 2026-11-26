"""Обрезка ленты сообщений OpenAI до лимита контекста."""

from __future__ import annotations

from typing import Any

from app.llm.context import content_chars, estimate_tokens


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
    messages: list[dict[str, Any]],
    max_length: int,
    *,
    max_chars: int | None = None,
    protect_tail: int = 0,
) -> list[dict[str, Any]]:
    """Оставляет не больше ``max_length`` сообщений, самые новые.

    Системный промпт в начале списка не вытесняется. После среза история
    выравнивается на ближайший ``user``: иначе посередине tool-раунда
    остаются сироты ``role: tool`` без ``tool_calls``, и ProxyAPI отвечает 400.
    Если в срезе нет ``user``, сохраняется весь ход, к которому относится
    хвост, даже длиннее лимита. Иначе длинный tool-раунд оставляет только
    system и стирает контекст темы.

    ``protect_tail`` — последние сообщения, которые не режутся (текущий ход).
    Хвост расширяется назад, чтобы не оторвать ``role: tool`` от ``tool_calls``.
    ``max_chars`` применяется после лимита по числу: режем с начала, по границе
    ``user``.
    """
    if max_length < 1:
        raise ValueError("max_length должен быть >= 1.")
    if protect_tail < 0:
        raise ValueError("protect_tail должен быть >= 0.")
    if max_chars is not None and max_chars < 1:
        raise ValueError("max_chars должен быть >= 1.")
    protected = _expanded_protect(messages, protect_tail)
    if protected == 0 and max_chars is None:
        return _trim_count(messages, max_length)
    if protected:
        head = messages[:-protected]
        tail = [dict(item) for item in messages[-protected:]]
    else:
        head = list(messages)
        tail = []
    room = max_length - len(tail)
    if room < 1:
        trimmed_head = (
            [dict(head[0])] if head and head[0].get("role") == "system" else []
        )
    else:
        trimmed_head = _trim_count(head, room)
    result = [*trimmed_head, *tail]
    if max_chars is None:
        return result
    return _trim_to_chars(
        result, max_chars, protect_n=_expanded_protect(result, protected)
    )


def shrink_to_token_limit(
    messages: list[dict[str, Any]],
    *,
    token_limit: int,
    chars_per_token: float,
    protect_tail: int = 0,
) -> list[dict[str, Any]]:
    """Срезает старые ходы, пока оценка не влезет в ``token_limit``.

    Ведущий ``system`` и хвост ``protect_tail`` не трогает. Если после этого
    оценка всё ещё больше лимита, возвращает то, что осталось.
    """
    current = [dict(item) for item in messages]
    while estimate_tokens(current, chars_per_token=chars_per_token) > token_limit:
        protect_n = _expanded_protect(current, protect_tail)
        nxt = _drop_oldest_turn(current, protect_n)
        if nxt is None:
            break
        current = nxt
    return current


def _trim_count(
    messages: list[dict[str, Any]], max_length: int
) -> list[dict[str, Any]]:
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


def _expanded_protect(messages: list[dict[str, Any]], protect_tail: int) -> int:
    """Хвост, который нельзя резать. ``tool`` подтягивает свой ``tool_calls``."""
    if protect_tail <= 0 or not messages:
        return 0
    start = max(0, len(messages) - protect_tail)
    while start > 0 and messages[start].get("role") == "tool":
        start -= 1
    return len(messages) - start


def _total_chars(messages: list[dict[str, Any]]) -> int:
    return sum(content_chars(item.get("content")) for item in messages)


def _trim_to_chars(
    messages: list[dict[str, Any]], max_chars: int, *, protect_n: int
) -> list[dict[str, Any]]:
    current = [dict(item) for item in messages]
    while _total_chars(current) > max_chars:
        nxt = _drop_oldest_turn(current, protect_n)
        if nxt is None:
            break
        current = nxt
    return current


def _drop_oldest_turn(
    messages: list[dict[str, Any]], protect_n: int
) -> list[dict[str, Any]] | None:
    """Снимает самый старый ход вне system и защищённого хвоста."""
    if protect_n >= len(messages):
        return None
    start = 1 if messages and messages[0].get("role") == "system" else 0
    end = len(messages) - protect_n
    if start >= end:
        return None
    if messages[start].get("role") != "user":
        drop_until = start + 1
    else:
        drop_until = start + 1
        while drop_until < end and messages[drop_until].get("role") != "user":
            drop_until += 1
    return [*messages[:start], *messages[drop_until:]]
