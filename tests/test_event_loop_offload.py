"""Тяжёлый PDF и PIL не должны занимать поток event loop."""

from __future__ import annotations

import io
import threading

import httpx
import pytest
from PIL import Image

from app.bot.handlers import media as media_mod
from app.llm.assist_ai import AssistAIClient


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), (1, 2, 3)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_pdf_render_runs_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop_thread = threading.get_ident()
    seen: list[int] = []

    def fake_pdf(data: bytes) -> list[bytes]:
        seen.append(threading.get_ident())
        assert data == b"%PDF"
        return [b"\x89PNG"]

    async def not_busy(message: object, state: object) -> bool:
        return False

    async def allow(user_id: int) -> bool:
        return True

    async def download(bot: object, file_id: str) -> bytes:
        return b"%PDF"

    async def vision(*args: object, **kwargs: object) -> None:
        return None

    monkeypatch.setattr(media_mod.R, "_reject_if_busy", not_busy)
    monkeypatch.setattr(media_mod.rate_limiter, "check_text_limit", allow)
    monkeypatch.setattr(media_mod, "download_telegram_file", download)
    monkeypatch.setattr(media_mod, "process_pdf", fake_pdf)
    monkeypatch.setattr(media_mod.R, "_process_vision_message", vision)

    class _User:
        id = 1

    class _Chat:
        id = 2

    class _Document:
        mime_type = "application/pdf"
        file_name = "doc.pdf"
        file_size = 100
        file_id = "f1"

    class _Message:
        document = _Document()
        bot = object()
        from_user = _User()
        chat = _Chat()
        caption = ""
        photo = None

        async def answer(self, text: str) -> None:
            raise AssertionError(text)

    await media_mod.handle_document(
        _Message(),  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
    )
    assert len(seen) == 1
    assert seen[0] != loop_thread


@pytest.mark.asyncio
async def test_edit_pil_runs_off_the_event_loop(
    assist_client: AssistAIClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import respx

    loop_thread = threading.get_ident()
    seen: list[int] = []
    real_size = assist_client._resolve_edit_size
    real_scale = assist_client._downscale_edit_input

    def size(explicit: str | None, data: bytes) -> str:
        seen.append(threading.get_ident())
        return real_size(explicit, data)

    def scale(data: bytes, mime: str) -> tuple[bytes, str]:
        seen.append(threading.get_ident())
        return real_scale(data, mime)

    monkeypatch.setattr(assist_client, "_resolve_edit_size", size)
    monkeypatch.setattr(assist_client, "_downscale_edit_input", scale)
    png = _png()
    with respx.mock(assert_all_called=False) as router:
        router.post("https://api.proxyapi.ru/v1/images/edits").mock(
            return_value=httpx.Response(
                200, json={"data": [{"url": "https://cdn.example/edited.png"}]}
            )
        )
        await assist_client.edit_image(png, "светлее")
        await assist_client.edit_images(
            [(png, "a.png", "image/png"), (png, "b.png", "image/png")],
            "светлее",
        )
    # edit_image: размер + один даунскейл; edit_images: размер + файл на каждый вход.
    assert len(seen) == 5
    assert all(ident != loop_thread for ident in seen)
