"""Tests for video poll/content URL host allowlist."""

from __future__ import annotations

import socket

import pytest

from app.llm.media_api import resolve_video_resource_url

PROVIDER = "https://api.proxyapi.ru/v1"
FALLBACK = f"{PROVIDER}/videos/abc"


def test_same_host_keeps_auth() -> None:
    url, with_auth = resolve_video_resource_url(
        f"{PROVIDER}/videos/abc/status",
        provider_base_url=PROVIDER,
        fallback=FALLBACK,
    )
    assert url.endswith("/status")
    assert with_auth is True


def test_same_host_http_falls_back() -> None:
    """Bearer must not ride plaintext even when the host matches the provider."""
    url, with_auth = resolve_video_resource_url(
        "http://api.proxyapi.ru/v1/videos/abc/content",
        provider_base_url=PROVIDER,
        fallback=FALLBACK,
    )
    assert url == FALLBACK
    assert with_auth is True


def test_public_cdn_https_without_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.llm.media_api.socket.getaddrinfo",
        lambda *_a, **_k: [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("8.8.8.8", 0))],
    )
    url, with_auth = resolve_video_resource_url(
        "https://cdn.example.com/out/video.mp4",
        provider_base_url=PROVIDER,
        fallback=FALLBACK,
    )
    assert url == "https://cdn.example.com/out/video.mp4"
    assert with_auth is False


def test_dns_rebinding_to_loopback_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.llm.media_api.socket.getaddrinfo",
        lambda *_a, **_k: [
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 0))
        ],
    )
    url, with_auth = resolve_video_resource_url(
        "https://evil.example/steal",
        provider_base_url=PROVIDER,
        fallback=FALLBACK,
    )
    assert url == FALLBACK
    assert with_auth is True


def test_metadata_ip_falls_back() -> None:
    url, with_auth = resolve_video_resource_url(
        "http://169.254.169.254/latest/meta-data/",
        provider_base_url=PROVIDER,
        fallback=FALLBACK,
    )
    assert url == FALLBACK
    assert with_auth is True


def test_localhost_falls_back() -> None:
    url, with_auth = resolve_video_resource_url(
        "http://127.0.0.1:8200/steal",
        provider_base_url=PROVIDER,
        fallback=FALLBACK,
    )
    assert url == FALLBACK
    assert with_auth is True


def test_evil_http_falls_back() -> None:
    url, with_auth = resolve_video_resource_url(
        "http://evil.example/steal",
        provider_base_url=PROVIDER,
        fallback=FALLBACK,
    )
    assert url == FALLBACK
    assert with_auth is True
