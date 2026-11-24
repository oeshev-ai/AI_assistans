"""Скачивание картинок с CDN: без редиректов и без приватных адресов."""

from __future__ import annotations

import socket

import httpx
import pytest
import respx

from app.llm.errors import AssistAIError
from app.llm.media_api import download_public_image

_TIMEOUT = httpx.Timeout(5.0, connect=2.0)


@pytest.mark.asyncio
async def test_private_literal_url_is_rejected() -> None:
    with pytest.raises(AssistAIError) as info:
        await download_public_image(
            "https://127.0.0.1/secret.png", timeout=_TIMEOUT, attempts=1
        )
    assert info.value.error_code == "GUARD_UNSAFE_URL"


@pytest.mark.asyncio
async def test_metadata_ip_is_rejected() -> None:
    with pytest.raises(AssistAIError) as info:
        await download_public_image(
            "http://169.254.169.254/latest/meta-data/",
            timeout=_TIMEOUT,
            attempts=1,
        )
    assert info.value.error_code == "GUARD_UNSAFE_URL"


@pytest.mark.asyncio
async def test_dns_to_private_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.llm.media_api.socket.getaddrinfo",
        lambda *_a, **_k: [
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 0))
        ],
    )
    with pytest.raises(AssistAIError) as info:
        await download_public_image(
            "https://evil.example/steal.png", timeout=_TIMEOUT, attempts=1
        )
    assert info.value.error_code == "GUARD_UNSAFE_URL"


@pytest.mark.asyncio
@respx.mock
async def test_redirect_is_not_followed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.llm.media_api.socket.getaddrinfo",
        lambda *_a, **_k: [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("8.8.8.8", 0))],
    )
    # Pin подменяет netloc на публичный IP; Host/SNI остаются cdn.example.
    route = respx.get("https://8.8.8.8/out.png").mock(
        return_value=httpx.Response(302, headers={"Location": "http://127.0.0.1/steal"})
    )
    with pytest.raises(AssistAIError):
        await download_public_image(
            "https://cdn.example/out.png", timeout=_TIMEOUT, attempts=1
        )
    assert route.called
    # Location не запрашивается: follow_redirects=False.
    assert respx.calls.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_public_cdn_bytes_returned(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.llm.media_api.socket.getaddrinfo",
        lambda *_a, **_k: [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("8.8.8.8", 0))],
    )
    respx.get("https://8.8.8.8/out.png").mock(
        return_value=httpx.Response(200, content=b"PNGDATA")
    )
    data = await download_public_image(
        "https://cdn.example/out.png", timeout=_TIMEOUT, attempts=1
    )
    assert data == b"PNGDATA"


@pytest.mark.asyncio
@respx.mock
async def test_oversized_cdn_image_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.llm.media_api.socket.getaddrinfo",
        lambda *_a, **_k: [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("8.8.8.8", 0))],
    )
    monkeypatch.setattr("app.llm.media_api.CDN_IMAGE_MAX_BYTES", 4)
    respx.get("https://8.8.8.8/out.png").mock(
        return_value=httpx.Response(200, content=b"PNGDATA")
    )
    with pytest.raises(AssistAIError) as info:
        await download_public_image(
            "https://cdn.example/out.png", timeout=_TIMEOUT, attempts=1
        )
    assert info.value.error_code == "PAYLOAD_TOO_LARGE"


@pytest.mark.asyncio
async def test_chunked_body_stops_above_the_cap() -> None:
    from app.llm.media_api import _read_capped_body

    class _Body:
        def __init__(self) -> None:
            self.headers = {"content-length": "3"}

        async def aiter_bytes(self):
            yield b"12345"
            yield b"67890"

    with pytest.raises(AssistAIError) as info:
        await _read_capped_body(_Body(), 4, "too big")  # type: ignore[arg-type]
    assert info.value.error_code == "PAYLOAD_TOO_LARGE"
