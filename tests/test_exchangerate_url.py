"""Currency client must use HTTPS for access_key transport."""

from __future__ import annotations

from app.llm.exchangerate import CONVERT_URL


def test_convert_url_is_https() -> None:
    assert CONVERT_URL.startswith("https://")
    assert "exchangerate.host/convert" in CONVERT_URL
