"""YC_S3_ENDPOINT не должен уводить запросы на внутренний адрес."""

from __future__ import annotations

import socket
from typing import Any

import pytest

from app.storage.s3_endpoint import (
    DEFAULT_S3_ENDPOINT,
    guard_s3_request,
    s3_endpoint_url,
    valid_bucket_name,
)
from app.storage.s3_storage import S3Config, S3StorageError, _build_client


def _public(*_args: Any, **_kwargs: Any) -> list[Any]:
    return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("77.88.55.88", 443))]


def _private(*_args: Any, **_kwargs: Any) -> list[Any]:
    return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("169.254.169.254", 443))]


def _config(endpoint: str = DEFAULT_S3_ENDPOINT) -> S3Config:
    return S3Config(
        endpoint=endpoint,
        region="ru-central1",
        bucket="media-bucket",
        access_key_id="key",
        secret_access_key="secret",
    )


def test_official_endpoint_is_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", _public)
    assert s3_endpoint_url("https://STORAGE.YANDEXCLOUD.NET/") == DEFAULT_S3_ENDPOINT
    assert s3_endpoint_url("") == DEFAULT_S3_ENDPOINT


@pytest.mark.parametrize(
    "url",
    [
        "https://169.254.169.254",
        "https://169.254.169.254/latest/meta-data",
        "https://127.0.0.1",
        "https://10.0.0.5",
        "https://storage.yandexcloud.net.evil.com",
        "https://evil.com",
        "http://storage.yandexcloud.net",
        "https://user:pass@storage.yandexcloud.net",
        "https://storage.yandexcloud.net:8443",
        "https://storage.yandexcloud.net/custom",
        "file:///etc/passwd",
        "https://storage.yandexcloud.net%2eevil.com",
    ],
)
def test_untrusted_endpoint_stops_the_process(
    monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("dns")),
    )
    with pytest.raises(SystemExit):
        s3_endpoint_url(url)


def test_private_resolution_of_official_host_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", _private)
    with pytest.raises(SystemExit):
        s3_endpoint_url(DEFAULT_S3_ENDPOINT)


def test_dns_failure_is_fatal_in_production(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")

    def _fail(*_args: Any, **_kwargs: Any) -> list[Any]:
        raise socket.gaierror("down")

    monkeypatch.setattr(socket, "getaddrinfo", _fail)
    with pytest.raises(SystemExit):
        s3_endpoint_url(DEFAULT_S3_ENDPOINT)


def test_dns_failure_outside_production_keeps_official_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENVIRONMENT", "development")

    def _fail(*_args: Any, **_kwargs: Any) -> list[Any]:
        raise socket.gaierror("down")

    monkeypatch.setattr(socket, "getaddrinfo", _fail)
    assert s3_endpoint_url(DEFAULT_S3_ENDPOINT) == DEFAULT_S3_ENDPOINT


def test_loopback_only_outside_production(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "development")
    assert s3_endpoint_url("http://127.0.0.1:9000") == "http://127.0.0.1:9000"
    monkeypatch.setenv("ENVIRONMENT", "production")
    with pytest.raises(SystemExit):
        s3_endpoint_url("http://127.0.0.1:9000")


def test_bucket_name_cannot_escape_the_host() -> None:
    assert valid_bucket_name("media-bucket")
    assert not valid_bucket_name("169.254.169.254")
    assert not valid_bucket_name("a..b")
    assert not valid_bucket_name("Bucket")


def test_request_to_metadata_is_not_sent() -> None:
    client = _build_client(_config())

    class _Request:
        url = "https://169.254.169.254/latest/meta-data"

    with pytest.raises(S3StorageError):
        client.meta.events.emit("before-send.s3.GetObject", request=_Request())


def test_request_stops_when_allowed_host_resolves_private(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", _private)
    with pytest.raises(S3StorageError):
        guard_s3_request(
            "https://media-bucket.storage.yandexcloud.net/uploads/a",
            allowed=frozenset(
                {"storage.yandexcloud.net", "media-bucket.storage.yandexcloud.net"}
            ),
        )
