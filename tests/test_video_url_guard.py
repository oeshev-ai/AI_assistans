"""Tests for video poll/content URL host allowlist and IP pinning."""

from __future__ import annotations

import socket

import pytest

from app.llm.media_api import _url_with_pinned_ip, resolve_video_resource_url

PROVIDER = "https://api.proxyapi.ru/v1"
FALLBACK = f"{PROVIDER}/videos/abc"


def test_same_host_keeps_auth() -> None:
    url, with_auth, pins = resolve_video_resource_url(
        f"{PROVIDER}/videos/abc/status",
        provider_base_url=PROVIDER,
        fallback=FALLBACK,
    )
    assert url.endswith("/status")
    assert with_auth is True
    assert pins == ()


def test_same_host_http_falls_back() -> None:
    """Bearer must not ride plaintext even when the host matches the provider."""
    url, with_auth, pins = resolve_video_resource_url(
        "http://api.proxyapi.ru/v1/videos/abc/content",
        provider_base_url=PROVIDER,
        fallback=FALLBACK,
    )
    assert url == FALLBACK
    assert with_auth is True
    assert pins == ()


def test_public_cdn_https_without_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.llm.media_api.socket.getaddrinfo",
        lambda *_a, **_k: [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("8.8.8.8", 0))],
    )
    url, with_auth, pins = resolve_video_resource_url(
        "https://cdn.example.com/out/video.mp4",
        provider_base_url=PROVIDER,
        fallback=FALLBACK,
    )
    assert url == "https://cdn.example.com/out/video.mp4"
    assert with_auth is False
    assert pins == ("8.8.8.8",)


def test_dns_rebinding_to_loopback_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.llm.media_api.socket.getaddrinfo",
        lambda *_a, **_k: [
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 0))
        ],
    )
    url, with_auth, pins = resolve_video_resource_url(
        "https://evil.example/steal",
        provider_base_url=PROVIDER,
        fallback=FALLBACK,
    )
    assert url == FALLBACK
    assert with_auth is True
    assert pins == ()


def test_mixed_public_private_dns_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.llm.media_api.socket.getaddrinfo",
        lambda *_a, **_k: [
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("8.8.8.8", 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 0)),
        ],
    )
    url, with_auth, pins = resolve_video_resource_url(
        "https://mixed.example/x",
        provider_base_url=PROVIDER,
        fallback=FALLBACK,
    )
    assert url == FALLBACK
    assert with_auth is True
    assert pins == ()


def test_metadata_ip_falls_back() -> None:
    url, with_auth, _pins = resolve_video_resource_url(
        "http://169.254.169.254/latest/meta-data/",
        provider_base_url=PROVIDER,
        fallback=FALLBACK,
    )
    assert url == FALLBACK
    assert with_auth is True


def test_localhost_falls_back() -> None:
    url, with_auth, _pins = resolve_video_resource_url(
        "http://127.0.0.1:8200/steal",
        provider_base_url=PROVIDER,
        fallback=FALLBACK,
    )
    assert url == FALLBACK
    assert with_auth is True


def test_evil_http_falls_back() -> None:
    url, with_auth, _pins = resolve_video_resource_url(
        "http://evil.example/steal",
        provider_base_url=PROVIDER,
        fallback=FALLBACK,
    )
    assert url == FALLBACK
    assert with_auth is True


def test_dns_failure_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: object, **_k: object) -> list[object]:
        raise socket.gaierror("name not resolved")

    monkeypatch.setattr("app.llm.media_api.socket.getaddrinfo", boom)
    url, with_auth, pins = resolve_video_resource_url(
        "https://unknown-cdn.example/video.mp4",
        provider_base_url=PROVIDER,
        fallback=FALLBACK,
    )
    assert url == FALLBACK
    assert with_auth is True
    assert pins == ()


def test_url_with_pinned_ip_keeps_path() -> None:
    assert (
        _url_with_pinned_ip("https://cdn.example.com/a/b?x=1", "1.2.3.4")
        == "https://1.2.3.4/a/b?x=1"
    )
    assert (
        _url_with_pinned_ip("https://cdn.example.com:8443/x", "2001:db8::1")
        == "https://[2001:db8::1]:8443/x"
    )
