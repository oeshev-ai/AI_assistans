"""Сборка REDIS_URL для Compose: пароль percent-encoding."""

from __future__ import annotations

from urllib.parse import quote


def compose_redis_url(
    host: str, password: str, *, port: int = 6379, db: int = 0
) -> str:
    """``redis://host:port/db`` или ``redis://:encoded@host`` если пароль задан.

    Пустой пароль не добавляет userinfo: ``redis://:@host`` ломает AUTH в arq.
    """
    target = (host or "redis").strip() or "redis"
    secret = password or ""
    if secret:
        return f"redis://:{quote(secret, safe='')}@{target}:{port}/{db}"
    return f"redis://{target}:{port}/{db}"
