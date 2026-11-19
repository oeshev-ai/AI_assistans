"""Tests for video poll/content URL host allowlist."""

from __future__ import annotations

from app.llm.assist_ai import resolve_video_resource_url

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


def test_public_cdn_https_without_auth() -> None:
    url, with_auth = resolve_video_resource_url(
        "https://cdn.example.com/out/video.mp4",
        provider_base_url=PROVIDER,
        fallback=FALLBACK,
    )
    assert url == "https://cdn.example.com/out/video.mp4"
    assert with_auth is False


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
