"""Видео через Polza /v1/media + fallback на ProxyAPI."""

from __future__ import annotations

import base64
import json
from collections.abc import Callable

import httpx
import pytest
import respx
import structlog
from prometheus_client import REGISTRY

from app.llm.assist_ai import AssistAIClient
from app.llm.errors import AssistAIError
from app.llm.media_api import VideoReference
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
    assert "generate_audio" not in body["input"]
    assert "images" not in body["input"]
    with_audio = build_video_body(
        model="bytedance/seedance-2",
        prompt="сцена",
        aspect_ratio="9:16",
        duration=10,
        resolution="480p",
        generate_audio=False,
    )
    assert with_audio["input"]["aspect_ratio"] == "9:16"
    assert with_audio["input"]["duration"] == "10"
    assert with_audio["input"]["resolution"] == "480p"
    assert with_audio["input"]["generate_audio"] is False
    with_images = build_video_body(
        model="bytedance/seedance-2",
        prompt="сцена",
        images=[{"type": "url", "data": "https://cdn.example/girl.png"}],
    )
    assert with_images["input"]["images"] == [
        {"type": "url", "data": "https://cdn.example/girl.png"}
    ]


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
        video = await client.generate_video(
            "кот бежит",
            duration=5,
            resolution="720p",
            aspect_ratio="9:16",
            generate_audio=True,
        )
    assert video.data == b"mp4-bytes"
    assert video.filename == "video.mp4"
    assert len(create.calls) == 1
    req = json.loads(create.calls[0].request.content.decode())
    assert req["model"] == "bytedance/seedance-2"
    assert req["input"]["duration"] == "5"
    assert req["input"]["aspect_ratio"] == "9:16"
    assert req["input"]["resolution"] == "720p"
    assert req["input"]["generate_audio"] is True
    assert "images" not in req["input"]
    assert status.called
    assert download.called
    assert not proxy.called


@pytest.mark.asyncio
async def test_generate_video_sends_reference_images(
    multi_client_factory: Callable[..., AssistAIClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Polza получает images, ProxyAPI при fallback — input_references."""
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
    refs = [
        VideoReference(url="https://cdn.example/girl.png"),
        VideoReference(data=b"png-bytes", mime_type="image/png"),
    ]
    with respx.mock(assert_all_called=False) as router:
        create = router.post(_POLZA_MEDIA).mock(
            return_value=httpx.Response(
                200,
                json={"id": "vid_ref", "status": "pending"},
            )
        )
        router.get(f"{_POLZA_MEDIA}/vid_ref").mock(
            return_value=httpx.Response(
                200,
                json={"id": "vid_ref", "status": "completed", "data": [{"url": _CDN}]},
            )
        )
        router.get(_CDN).mock(return_value=httpx.Response(200, content=b"mp4"))
        await client.generate_video("девушка с первой картинки", references=refs)
    body = json.loads(create.calls[0].request.content.decode())
    images = body["input"]["images"]
    assert images[0] == {"type": "url", "data": "https://cdn.example/girl.png"}
    assert images[1]["type"] == "base64"
    assert base64.b64decode(images[1]["data"]) == b"png-bytes"


@pytest.mark.asyncio
async def test_generate_video_proxy_fallback_keeps_references(
    multi_client_factory: Callable[..., AssistAIClient],
) -> None:
    client = multi_client_factory(
        routes={"video": "polza", "image": "polza"},
        video_fallback="proxyapi",
    )
    refs = [VideoReference(url="https://cdn.example/girl.png")]
    with respx.mock(assert_all_called=False) as router:
        router.post(_POLZA_MEDIA).mock(return_value=httpx.Response(503))
        create = router.post(_PROXY_VIDEOS).mock(
            return_value=httpx.Response(
                200, json={"id": "job_ref", "status": "completed"}
            )
        )
        router.get(f"{_PROXY_VIDEOS}/job_ref/content").mock(
            return_value=httpx.Response(200, content=b"proxy-mp4")
        )
        video = await client.generate_video("сцена", references=refs)
    assert video.data == b"proxy-mp4"
    payload = json.loads(create.calls[0].request.content.decode())
    assert payload["input_references"] == [
        {
            "type": "image_url",
            "image_url": {"url": "https://cdn.example/girl.png"},
        }
    ]


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
async def test_generate_video_fallback_keeps_persona_settings(
    multi_client_factory: Callable[..., AssistAIClient],
) -> None:
    """И Polza, и запасной ProxyAPI получают настройки ролика персоны."""
    client = multi_client_factory(
        routes={"video": "polza", "image": "polza"},
        video_fallback="proxyapi",
    )
    with respx.mock(assert_all_called=False) as router:
        polza = router.post(_POLZA_MEDIA).mock(return_value=httpx.Response(503))
        create = router.post(_PROXY_VIDEOS).mock(
            return_value=httpx.Response(
                200, json={"id": "job_opts", "status": "completed"}
            )
        )
        router.get(f"{_PROXY_VIDEOS}/job_opts/content").mock(
            return_value=httpx.Response(200, content=b"proxy-mp4")
        )
        video = await client.generate_video(
            "сцена",
            aspect_ratio="9:16",
            duration=12,
            resolution="480p",
            generate_audio=True,
        )
    assert video.data == b"proxy-mp4"
    polza_body = json.loads(polza.calls[0].request.content.decode())
    assert polza_body["input"]["aspect_ratio"] == "9:16"
    assert polza_body["input"]["duration"] == "12"
    assert polza_body["input"]["resolution"] == "480p"
    assert polza_body["input"]["generate_audio"] is True
    proxy_body = json.loads(create.calls[0].request.content.decode())
    assert proxy_body["aspect_ratio"] == "9:16"
    assert proxy_body["duration"] == 12
    assert proxy_body["resolution"] == "480p"
    assert proxy_body["generate_audio"] is True


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
    assert info.value.pending_provider == "polza"
    assert info.value.pending_job_id == "vid_stuck"
    assert info.value.pending_kind == "video"
    assert "видео" in info.value.message.casefold()
    assert len(polza.calls) == 1
    assert not proxy.called


@pytest.mark.asyncio
async def test_proxyapi_poll_survives_connection_blip(
    multi_client_factory: Callable[..., AssistAIClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Оплаченная задача ProxyAPI не бросается на первом обрыве опроса."""
    monkeypatch.setattr("app.llm.media_api._VIDEO_POLL_INTERVAL", 0.0)
    monkeypatch.setattr("app.llm.media_api._VIDEO_RETRY_PAUSE", 0.0)
    client = multi_client_factory(
        routes={"video": "polza", "image": "polza"},
        video_fallback="proxyapi",
    )
    status_url = f"{_PROXY_VIDEOS}/job_retry"
    with respx.mock(assert_all_called=False) as router:
        router.post(_POLZA_MEDIA).mock(return_value=httpx.Response(503))
        create = router.post(_PROXY_VIDEOS).mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "job_retry",
                    "status": "pending",
                    "polling_url": status_url,
                },
            )
        )
        router.get(status_url).mock(
            side_effect=[
                httpx.ConnectError("All connection attempts failed"),
                httpx.Response(200, json={"id": "job_retry", "status": "completed"}),
            ]
        )
        router.get(f"{status_url}/content").mock(
            return_value=httpx.Response(200, content=b"ready-mp4")
        )
        video = await client.generate_video("сцена")
    assert video.data == b"ready-mp4"
    assert len(create.calls) == 1


async def test_generate_video_download_fail_no_fallback(
    multi_client_factory: Callable[..., AssistAIClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.llm.media_api._VIDEO_RETRY_PAUSE", 0.0)
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
    assert info.value.pending_url is None
    assert not proxy.called


@pytest.mark.asyncio
async def test_polza_video_downloads_from_cdn_mirror(
    multi_client_factory: Callable[..., AssistAIClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """s3.polza.ai часто не открывается: тот же объект берётся с зеркала."""
    monkeypatch.setattr("app.llm.media_api._VIDEO_RETRY_PAUSE", 0.0)
    monkeypatch.setattr(
        "app.llm.media_api.resolve_video_resource_url",
        lambda candidate, **_k: (candidate or "", False, ()),
    )
    primary = "https://s3.polza.ai/media/clip.mp4"
    mirror = "https://s3-alt.polza.ai/media/clip.mp4"
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
                    "id": "vid_mirror",
                    "status": "completed",
                    "data": [{"url": primary}],
                },
            )
        )
        mirrored = router.get(mirror).mock(
            return_value=httpx.Response(200, content=b"from-alt")
        )
        proxy = router.post(_PROXY_VIDEOS).mock(return_value=httpx.Response(200))
        video = await client.generate_video("сцена")
    assert video.data == b"from-alt"
    assert mirrored.called
    assert not proxy.called


@pytest.mark.asyncio
async def test_polza_video_download_keeps_cdn_url(
    multi_client_factory: Callable[..., AssistAIClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.llm.media_api._VIDEO_RETRY_PAUSE", 0.0)
    monkeypatch.setattr(
        "app.llm.media_api.resolve_video_resource_url",
        lambda candidate, **_k: (candidate or "", False, ()),
    )
    primary = "https://s3.polza.ai/media/clip.mp4"
    mirror = "https://s3-alt.polza.ai/media/clip.mp4"
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
                    "id": "vid_keep",
                    "status": "completed",
                    "data": [{"url": primary}],
                },
            )
        )
        router.get(mirror).mock(side_effect=httpx.ConnectError("boom"))
        router.get(primary).mock(side_effect=httpx.ConnectError("boom"))
        proxy = router.post(_PROXY_VIDEOS).mock(return_value=httpx.Response(200))
        with pytest.raises(AssistAIError) as info:
            await client.generate_video("сцена")
    assert info.value.pending_url == primary
    assert info.value.pending_provider == "polza"
    assert info.value.pending_job_id == "vid_keep"
    assert info.value.job_committed is True
    assert "://" not in info.value.message
    assert not proxy.called


@pytest.mark.asyncio
async def test_proxy_video_download_keeps_the_job(
    multi_client_factory: Callable[..., AssistAIClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Оплаченная задача ProxyAPI остаётся в очереди, если файл не скачался."""
    monkeypatch.setattr("app.llm.media_api._VIDEO_POLL_INTERVAL", 0.0)
    monkeypatch.setattr("app.llm.media_api._VIDEO_RETRY_PAUSE", 0.0)
    monkeypatch.setattr("app.llm.media_api._VIDEO_POLL_MAX_SECONDS", 5.0)
    client = multi_client_factory(
        routes={"video": "proxyapi", "image": "proxyapi"},
    )
    status_url = f"{_PROXY_VIDEOS}/job_paid"
    with respx.mock(assert_all_called=False) as router:
        create = router.post(_PROXY_VIDEOS).mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "job_paid",
                    "status": "completed",
                    "polling_url": status_url,
                },
            )
        )
        router.get(f"{status_url}/content").mock(side_effect=httpx.ConnectError("boom"))
        with pytest.raises(AssistAIError) as info:
            await client.generate_video("сцена")
    assert info.value.pending_provider == "proxyapi"
    assert info.value.pending_job_id == "job_paid"
    assert info.value.pending_kind == "video"
    assert info.value.job_committed is True
    assert len(create.calls) == 1
