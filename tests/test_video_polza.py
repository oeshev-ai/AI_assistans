"""Видео через Polza /v1/media + fallback на ProxyAPI."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest
import respx
import structlog
from prometheus_client import REGISTRY

from app.llm.assist_ai import AssistAIClient
from app.llm.errors import AssistAIError
from app.llm.models_config import resolve_polza_video_model
from app.llm.polza_media import build_video_body

_POLZA_MEDIA = "https://polza.ai/api/v1/media"
_PROXY_VIDEOS = "https://api.proxyapi.ru/v1/videos"
_CDN = "https://cdn.example/out.mp4"


def _fallback_count(source: str, target: str, reason: str) -> float:
    value = REGISTRY.get_sample_value(
        "provider_fallback_total", {"from": source, "to": target, "reason": reason}
    )
    return value or 0.0


def test_resolve_polza_video_model_alias() -> None:
    assert resolve_polza_video_model("bytedance/seedance-2.0") == "bytedance/seedance-2"
    assert resolve_polza_video_model("bytedance/seedance-2") == "bytedance/seedance-2"


def test_build_video_body_seedance() -> None:
    body = build_video_body(
        model="bytedance/seedance-2",
        prompt="кот бежит",
        duration=8,
        resolution="720p",
    )
    assert body["model"] == "bytedance/seedance-2"
    assert body["async"] is True
    assert body["input"]["prompt"] == "кот бежит"
    assert body["input"]["duration"] == "8"
    assert body["input"]["resolution"] == "720p"
    assert body["input"]["aspect_ratio"] == "16:9"


@pytest.mark.asyncio
async def test_generate_video_polza_success(
    multi_client_factory: Callable[..., AssistAIClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.llm.media_api.resolve_video_resource_url",
        lambda candidate, **_k: (candidate or _CDN, False, ()),
    )
    client = multi_client_factory(
        routes={"video": "polza", "image": "polza"},
        video_fallback="proxyapi",
        polza_total_wait=8.0,
        polza_poll_interval=0.01,
    )
    with respx.mock(assert_all_called=False) as router:
        create = router.post(_POLZA_MEDIA).mock(
            return_value=httpx.Response(
                200,
                json={"id": "vid_1", "status": "pending", "object": "media.generation"},
            )
        )
        status = router.get(f"{_POLZA_MEDIA}/vid_1").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "vid_1",
                    "status": "completed",
                    "data": [{"url": _CDN}],
                },
            )
        )
        download = router.get(_CDN).mock(
            return_value=httpx.Response(200, content=b"mp4-bytes")
        )
        proxy = router.post(_PROXY_VIDEOS).mock(return_value=httpx.Response(200))
        video = await client.generate_video("кот бежит", duration=5, resolution="720p")
    assert video.data == b"mp4-bytes"
    assert video.filename == "video.mp4"
    assert len(create.calls) == 1
    req = json.loads(create.calls[0].request.content.decode())
    assert req["model"] == "bytedance/seedance-2"
    assert req["input"]["duration"] == "5"
    assert status.called
    assert download.called
    assert not proxy.called


@pytest.mark.asyncio
async def test_generate_video_polza_rejects_private_content_url(
    multi_client_factory: Callable[..., AssistAIClient],
) -> None:
    """Unsafe CDN/content URL must not be fetched (and must not send Bearer)."""
    client = multi_client_factory(
        routes={"video": "polza", "image": "polza"},
        video_fallback="proxyapi",
        polza_total_wait=8.0,
        polza_poll_interval=0.01,
    )
    evil = "http://127.0.0.1:9/steal"
    with respx.mock(assert_all_called=False) as router:
        router.post(_POLZA_MEDIA).mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "vid_bad",
                    "status": "pending",
                    "object": "media.generation",
                },
            )
        )
        router.get(f"{_POLZA_MEDIA}/vid_bad").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "vid_bad",
                    "status": "completed",
                    "data": [{"url": evil}],
                },
            )
        )
        evil_get = router.get(evil).mock(return_value=httpx.Response(200, content=b"x"))
        proxy = router.post(_PROXY_VIDEOS).mock(return_value=httpx.Response(200))
        with pytest.raises(AssistAIError) as info:
            await client.generate_video("сцена")
    assert info.value.error_code == "GUARD_UNSAFE_URL"
    assert not evil_get.called
    assert not proxy.called


@pytest.mark.asyncio
async def test_generate_video_falls_back_to_proxyapi(
    multi_client_factory: Callable[..., AssistAIClient],
) -> None:
    client = multi_client_factory(
        routes={"video": "polza", "image": "polza"},
        video_fallback="proxyapi",
    )
    before = _fallback_count("polza", "proxyapi", "service_unavailable")
    with respx.mock(assert_all_called=False) as router:
        polza = router.post(_POLZA_MEDIA).mock(return_value=httpx.Response(503))
        create = router.post(_PROXY_VIDEOS).mock(
            return_value=httpx.Response(
                200, json={"id": "job_1", "status": "completed"}
            )
        )
        content = router.get(f"{_PROXY_VIDEOS}/job_1/content").mock(
            return_value=httpx.Response(200, content=b"proxy-mp4")
        )
        with structlog.testing.capture_logs() as logs:
            video = await client.generate_video("сцена")
    assert video.data == b"proxy-mp4"
    assert len(polza.calls) == 1
    assert len(create.calls) == 1
    assert content.called
    assert _fallback_count("polza", "proxyapi", "service_unavailable") == before + 1
    event = next(item for item in logs if item["event"] == "provider_fallback")
    assert (event["from"], event["to"], event["reason"]) == (
        "polza",
        "proxyapi",
        "service_unavailable",
    )


@pytest.mark.asyncio
async def test_generate_video_client_error_no_fallback(
    multi_client_factory: Callable[..., AssistAIClient],
) -> None:
    client = multi_client_factory(
        routes={"video": "polza", "image": "polza"},
        video_fallback="proxyapi",
    )
    with respx.mock(assert_all_called=False) as router:
        polza = router.post(_POLZA_MEDIA).mock(
            return_value=httpx.Response(400, json={"error": {"code": "BAD_REQUEST"}})
        )
        proxy = router.post(_PROXY_VIDEOS).mock(return_value=httpx.Response(200))
        with pytest.raises(AssistAIError) as info:
            await client.generate_video("сцена")
    assert info.value.status_code == 400
    assert len(polza.calls) == 1
    assert not proxy.called


@pytest.mark.asyncio
async def test_generate_video_poll_timeout_no_fallback(
    multi_client_factory: Callable[..., AssistAIClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After Polza accepts the job, poll timeout must not open ProxyAPI."""
    monkeypatch.setattr("app.llm.media_api._VIDEO_POLL_MAX_SECONDS", 0.05)
    client = multi_client_factory(
        routes={"video": "polza", "image": "polza"},
        video_fallback="proxyapi",
        polza_total_wait=0.05,
        polza_poll_interval=0.01,
    )
    with respx.mock(assert_all_called=False) as router:
        polza = router.post(_POLZA_MEDIA).mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "vid_stuck",
                    "status": "pending",
                    "object": "media.generation",
                },
            )
        )
        router.get(f"{_POLZA_MEDIA}/vid_stuck").mock(
            return_value=httpx.Response(
                200,
                json={"id": "vid_stuck", "status": "pending"},
            )
        )
        proxy = router.post(_PROXY_VIDEOS).mock(return_value=httpx.Response(200))
        with pytest.raises(AssistAIError) as info:
            await client.generate_video("сцена")
    assert info.value.error_code == "POLL_TIMEOUT"
    assert info.value.job_committed is True
    assert "видео" in info.value.message.casefold()
    assert len(polza.calls) == 1
    assert not proxy.called


@pytest.mark.asyncio
async def test_generate_video_download_fail_no_fallback(
    multi_client_factory: Callable[..., AssistAIClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.llm.media_api.resolve_video_resource_url",
        lambda candidate, **_k: (candidate or _CDN, False, ()),
    )
    client = multi_client_factory(
        routes={"video": "polza", "image": "polza"},
        video_fallback="proxyapi",
        polza_total_wait=8.0,
        polza_poll_interval=0.01,
    )
    with respx.mock(assert_all_called=False) as router:
        router.post(_POLZA_MEDIA).mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "vid_dl",
                    "status": "pending",
                    "object": "media.generation",
                },
            )
        )
        router.get(f"{_POLZA_MEDIA}/vid_dl").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "vid_dl",
                    "status": "completed",
                    "data": [{"url": _CDN}],
                },
            )
        )
        router.get(_CDN).mock(side_effect=httpx.ConnectError("boom"))
        proxy = router.post(_PROXY_VIDEOS).mock(return_value=httpx.Response(200))
        with pytest.raises(AssistAIError) as info:
            await client.generate_video("сцена")
    assert info.value.error_code == "NETWORK_ERROR"
    assert info.value.job_committed is True
    assert not proxy.called
