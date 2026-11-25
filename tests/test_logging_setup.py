"""printf-аргументы structlog попадают в текст события, а не в positional_args."""

from __future__ import annotations

import json
import logging

import structlog

from app.infra.logging_setup import configure_logging


def _json_records(raw: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("{"):
            records.append(json.loads(line))
    return records


def test_percent_s_is_interpolated_into_json_event(capsys) -> None:
    root = logging.getLogger()
    previous_handlers = list(root.handlers)
    previous_level = root.level
    configure_logging()
    try:
        log = structlog.get_logger("test.positional.arguments")
        log.info("Redis: активная тема chat_id=%s topic_id=%s", 7, "t-1")
        log.info("state", chat_id=7)
        log.warning("PDF: страница %s слишком большая (%.0f px)", 2, 1500.4)
        logging.getLogger("test.foreign.arguments").info("file %s", "a.txt")
        out = capsys.readouterr().out
    finally:
        root.handlers.clear()
        for handler in previous_handlers:
            root.addHandler(handler)
        root.setLevel(previous_level)

    records = _json_records(out)
    ours = [
        item
        for item in records
        if item.get("logger") == "test.positional.arguments"
    ]
    assert [item["event"] for item in ours] == [
        "Redis: активная тема chat_id=7 topic_id=t-1",
        "state",
        "PDF: страница 2 слишком большая (1500 px)",
    ]
    assert all("positional_args" not in item for item in ours)
    assert ours[1]["chat_id"] == 7
    foreign = next(
        item for item in records if item.get("logger") == "test.foreign.arguments"
    )
    assert foreign["event"] == "file a.txt"
    assert "positional_args" not in foreign
