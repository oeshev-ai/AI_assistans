"""Redis backup must not put the password on the redis-cli command line."""

from __future__ import annotations

from pathlib import Path


def test_backup_redis_sh_uses_env_auth() -> None:
    text = Path("scripts/backup_redis.sh").read_text(encoding="utf-8")
    assert "REDISCLI_AUTH" in text
    assert "-a " not in text
    assert "-a\t" not in text
