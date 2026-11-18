"""В истории темы не остаётся URL и base64 медиа-инструментов."""

from __future__ import annotations

import json

from app.bot.runtime import _compact_tool_content


def test_edit_result_drops_url_and_base64() -> None:
    raw = json.dumps(
        {
            "status": "success",
            "image_count": 1,
            "url": "https://cdn.example/secret.png",
            "b64_json": "A" * 120,
        },
        ensure_ascii=False,
    )
    compacted = json.loads(_compact_tool_content(raw))
    assert "url" not in compacted
    assert "b64_json" not in compacted
    assert compacted["status"] == "success"
    assert "отправлено" in compacted["note"]
    again = json.dumps(compacted, ensure_ascii=False)
    assert "https://" not in again
    assert "AAAA" not in again
