"""Allowlist parsing and access decisions."""

from __future__ import annotations

from app.infra.access import is_production, is_user_allowed, parse_allowed_user_ids


def test_parse_empty_is_open_mode() -> None:
    assert parse_allowed_user_ids("") == frozenset()
    assert parse_allowed_user_ids("   ") == frozenset()


def test_parse_comma_and_space_separated() -> None:
    assert parse_allowed_user_ids("111, 222;333") == frozenset({111, 222, 333})
    assert parse_allowed_user_ids("42 43") == frozenset({42, 43})


def test_parse_skips_invalid_tokens() -> None:
    assert parse_allowed_user_ids("1,abc,2") == frozenset({1, 2})


def test_open_mode_allows_anyone() -> None:
    allowed: frozenset[int] = frozenset()
    assert is_user_allowed(1, allowed) is True
    assert is_user_allowed(None, allowed) is True


def test_closed_mode_requires_membership() -> None:
    allowed: frozenset[int] = frozenset({10, 20})
    assert is_user_allowed(10, allowed) is True
    assert is_user_allowed(99, allowed) is False
    assert is_user_allowed(None, allowed) is False


def test_is_production(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("ENVIRONMENT", "production")
    assert is_production() is True
    monkeypatch.setenv("ENVIRONMENT", "development")
    assert is_production() is False
