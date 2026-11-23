"""Адрес Yandex Object Storage, с которого процесс не уходит на чужой хост."""

from __future__ import annotations

import ipaddress
import re
import socket
from typing import NoReturn
from urllib.parse import urlsplit

import structlog

logger = structlog.get_logger("s3_endpoint")

YANDEX_S3_HOST = "storage.yandexcloud.net"
DEFAULT_S3_ENDPOINT = f"https://{YANDEX_S3_HOST}"
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_BUCKET_RE = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")


def valid_bucket_name(name: str) -> bool:
    """Имя бакета не должно менять хост virtual-hosted запроса."""
    if not _BUCKET_RE.fullmatch(name) or ".." in name:
        return False
    try:
        ipaddress.ip_address(name)
    except ValueError:
        return True
    return False


def s3_endpoint_url(url: str, *, default: str = DEFAULT_S3_ENDPOINT) -> str:
    """Нормализованный endpoint или остановка процесса.

    В production допустим только ``https://storage.yandexcloud.net``.
    Вне production дополнительно можно указать loopback (локальный S3).
    """
    cleaned = (url or "").strip() or default
    if any(char in cleaned for char in " \t\r\n\\@"):
        _reject(None)
    try:
        parts = urlsplit(cleaned)
        host = parts.hostname
        port = parts.port
    except ValueError:
        _reject(None)
    if host is None or parts.username or parts.password:
        _reject(host)
    try:
        host_idna = host.encode("idna").decode("ascii")
    except UnicodeError:
        _reject(host)
    if parts.query or parts.fragment or parts.path not in ("", "/"):
        _reject(host_idna)

    if host_idna == YANDEX_S3_HOST and parts.scheme == "https" and port in (None, 443):
        _assert_resolved(host_idna, loopback=False)
        return DEFAULT_S3_ENDPOINT

    from app.infra.access import is_production

    if (
        not is_production()
        and host_idna in _LOOPBACK_HOSTS
        and parts.scheme in {"http", "https"}
    ):
        _assert_resolved(host_idna, loopback=True)
        return _format(parts.scheme, host_idna, port)

    _reject(host_idna)


def allowed_s3_hosts(endpoint: str, bucket: str) -> frozenset[str]:
    """Хосты, на которые этому endpoint разрешено слать запросы."""
    host = urlsplit(endpoint).hostname or ""
    if host == YANDEX_S3_HOST:
        return frozenset({host, f"{bucket}.{host}"})
    return frozenset({host})


def guard_s3_request(url: str, *, allowed: frozenset[str]) -> None:
    """Отклоняет URL, если хост не из списка или резолвится во внутренний адрес."""
    try:
        parts = urlsplit(url)
        host = parts.hostname
    except ValueError:
        host = None
    if (
        host is None
        or host not in allowed
        or parts.username
        or parts.password
        or parts.scheme not in {"https", "http"}
    ):
        logger.error("s3_request_host_rejected", host=host or "invalid")
        from app.storage.s3_storage import S3StorageError

        raise S3StorageError("Запрос к S3 ушёл на неразрешённый хост")
    _assert_resolved(host, loopback=host in _LOOPBACK_HOSTS, fatal=False)


def _format(scheme: str, host: str, port: int | None) -> str:
    shown = f"[{host}]" if ":" in host else host
    if (
        port is None
        or (scheme == "https" and port == 443)
        or (scheme == "http" and port == 80)
    ):
        return f"{scheme}://{shown}"
    return f"{scheme}://{shown}:{port}"


def _reject(host: str | None) -> NoReturn:
    logger.error(
        "s3_endpoint_rejected",
        name="YC_S3_ENDPOINT",
        host=host or "invalid",
        hint="Use https://storage.yandexcloud.net",
    )
    raise SystemExit(1)


def _assert_resolved(host: str, *, loopback: bool, fatal: bool = True) -> None:
    port = 443 if not loopback else 80
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        from app.infra.access import is_production

        if not fatal or is_production():
            logger.error("s3_endpoint_dns_failed", host=host, error=type(exc).__name__)
            if fatal:
                raise SystemExit(1) from exc
            from app.storage.s3_storage import S3StorageError

            raise S3StorageError("Не удалось разрешить адрес S3") from exc
        logger.warning("s3_endpoint_dns_failed", host=host, error=type(exc).__name__)
        return
    addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for info in infos:
        raw = info[4][0]
        # При сборке без IPv6 typeshed даёт sockaddr как (int, bytes).
        if not isinstance(raw, str):
            addresses.clear()
            break
        addresses.append(_address(raw))
    safe = _is_loopback if loopback else _is_public
    if not addresses or any(not safe(ip) for ip in addresses):
        logger.error("s3_endpoint_private_address", host=host)
        if fatal:
            raise SystemExit(1)
        from app.storage.s3_storage import S3StorageError

        raise S3StorageError("Адрес S3 резолвится во внутреннюю сеть")


def _address(raw: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    return ipaddress.ip_address(raw.split("%", 1)[0])


def _is_public(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        ip = mapped
    return bool(ip.is_global)


def _is_loopback(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        ip = mapped
    return bool(ip.is_loopback)
