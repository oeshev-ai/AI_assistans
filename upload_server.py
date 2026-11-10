"""HTTP: страница загрузки + presign/complete для Yandex Object Storage."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable

import structlog
from aiohttp import web

from s3_storage import S3Storage, S3StorageError
from upload_sessions import UploadSession, UploadSessionStore, object_key_for

logger = structlog.get_logger("upload_server")

UploadCompleteHandler = Callable[[UploadSession], Awaitable[None]]

DEFAULT_ALLOWED_MIME = (
    "audio/mpeg",
    "audio/mp3",
    "audio/mp4",
    "audio/x-m4a",
    "audio/m4a",
    "audio/wav",
    "audio/x-wav",
    "audio/ogg",
    "audio/webm",
    "video/mp4",
    "video/webm",
    "video/quicktime",
    "video/x-matroska",
)

ALLOWED_EXTENSIONS = (
    ".mp3",
    ".m4a",
    ".wav",
    ".ogg",
    ".oga",
    ".webm",
    ".mp4",
    ".mov",
    ".mkv",
)


def _allowed_mime_types() -> set[str]:
    raw = os.getenv("UPLOAD_ALLOWED_MIME_TYPES", "").strip()
    if not raw:
        return set(DEFAULT_ALLOWED_MIME)
    return {item.strip() for item in raw.split(",") if item.strip()}


def _extract_token(request: web.Request) -> str:
    token = request.rel_url.query.get("t", "").strip()
    if token:
        return token
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return (request.headers.get("X-Upload-Token") or "").strip()


def _page_html(upload_id: str, token: str, max_mb: int) -> str:
    allowed = ", ".join(ALLOWED_EXTENSIONS)
    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Загрузка аудио/видео</title>
  <style>
    :root {{
      --bg: #0f1419;
      --card: #1a2332;
      --text: #e7ecf3;
      --muted: #8b9bb4;
      --accent: #3d8bfd;
      --ok: #3ecf8e;
      --err: #ff6b6b;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; min-height: 100vh; font-family: "Segoe UI", system-ui, sans-serif;
      background: radial-gradient(1200px 600px at 10% -10%, #1c2a44, var(--bg));
      color: var(--text); display: grid; place-items: center; padding: 24px;
    }}
    .card {{
      width: min(440px, 100%); background: color-mix(in srgb, var(--card) 92%, black);
      border: 1px solid #2a3a52; border-radius: 16px; padding: 28px 24px;
      box-shadow: 0 20px 50px rgba(0,0,0,.35);
    }}
    h1 {{ margin: 0 0 8px; font-size: 1.35rem; font-weight: 650; }}
    p {{ margin: 0 0 18px; color: var(--muted); line-height: 1.45; font-size: .95rem; }}
    input[type=file] {{
      width: 100%; padding: 12px; border-radius: 10px; border: 1px dashed #3a4d6b;
      background: #121a26; color: var(--text); margin-bottom: 14px;
    }}
    button {{
      width: 100%; border: 0; border-radius: 10px; padding: 12px 16px;
      background: var(--accent); color: white; font-weight: 600; font-size: 1rem;
      cursor: pointer;
    }}
    button:disabled {{ opacity: .55; cursor: not-allowed; }}
    .bar {{
      display: none; height: 8px; background: #243247; border-radius: 999px;
      overflow: hidden; margin: 14px 0 8px;
    }}
    .bar > span {{
      display: block; height: 100%; width: 0; background: linear-gradient(90deg,#3d8bfd,#6ea8fe);
      transition: width .15s linear;
    }}
    .status {{ min-height: 1.3em; font-size: .92rem; margin-top: 10px; }}
    .status.ok {{ color: var(--ok); }}
    .status.err {{ color: var(--err); }}
  </style>
</head>
<body>
  <div class="card">
    <h1>Большой файл для расшифровки</h1>
    <p>Выберите аудио или видео (до {max_mb} МБ). Допустимо: {allowed}. Файл уйдёт напрямую в облако, минуя лимит Telegram.</p>
    <input id="file" type="file" accept="{','.join(ALLOWED_EXTENSIONS)},audio/*,video/*"/>
    <button id="btn" type="button">Загрузить</button>
    <div class="bar" id="bar"><span id="fill"></span></div>
    <div class="status" id="status"></div>
  </div>
  <script>
    const UPLOAD_ID = {upload_id!r};
    const TOKEN = {token!r};
    const MAX_MB = {max_mb};
    const ALLOWED_EXT = {list(ALLOWED_EXTENSIONS)!r};

    const fileInput = document.getElementById('file');
    const btn = document.getElementById('btn');
    const bar = document.getElementById('bar');
    const fill = document.getElementById('fill');
    const statusEl = document.getElementById('status');

    function setStatus(text, kind) {{
      statusEl.textContent = text || '';
      statusEl.className = 'status' + (kind ? ' ' + kind : '');
    }}

    function extOk(name) {{
      const lower = (name || '').toLowerCase();
      return ALLOWED_EXT.some((ext) => lower.endsWith(ext));
    }}

    btn.addEventListener('click', async () => {{
      const file = fileInput.files && fileInput.files[0];
      if (!file) {{ setStatus('Выберите файл', 'err'); return; }}
      if (!extOk(file.name)) {{
        setStatus('Неподдерживаемый формат файла', 'err');
        return;
      }}
      if (file.size > MAX_MB * 1024 * 1024) {{
        setStatus('Файл больше лимита ' + MAX_MB + ' МБ', 'err');
        return;
      }}

      btn.disabled = true;
      bar.style.display = 'block';
      fill.style.width = '0%';
      setStatus('Получаю ссылку для загрузки…');

      try {{
        const mime = file.type || 'application/octet-stream';
        const presignRes = await fetch('/api/uploads/' + UPLOAD_ID + '/presign?t=' + encodeURIComponent(TOKEN), {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json', 'X-Upload-Token': TOKEN }},
          body: JSON.stringify({{ fileName: file.name, fileType: mime, fileSize: file.size }})
        }});
        const presign = await presignRes.json();
        if (!presignRes.ok) throw new Error(presign.error || 'Не удалось получить ссылку');

        setStatus('Загружаю в облако…');
        await new Promise((resolve, reject) => {{
          const xhr = new XMLHttpRequest();
          xhr.open('PUT', presign.signedUrl);
          xhr.setRequestHeader('Content-Type', mime);
          xhr.upload.onprogress = (e) => {{
            if (e.lengthComputable) fill.style.width = Math.round(e.loaded / e.total * 100) + '%';
          }};
          xhr.onload = () => {{
            if (xhr.status >= 200 && xhr.status < 300) resolve();
            else reject(new Error('Ошибка загрузки в бакет (' + xhr.status + ')'));
          }};
          xhr.onerror = () => reject(new Error('Сеть: не удалось загрузить файл'));
          xhr.send(file);
        }});

        setStatus('Подтверждаю загрузку…');
        const completeRes = await fetch('/api/uploads/' + UPLOAD_ID + '/complete?t=' + encodeURIComponent(TOKEN), {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json', 'X-Upload-Token': TOKEN }},
          body: JSON.stringify({{
            objectKey: presign.objectKey,
            fileName: file.name,
            fileType: mime,
            fileSize: file.size
          }})
        }});
        const complete = await completeRes.json();
        if (!completeRes.ok) throw new Error(complete.error || 'Ошибка подтверждения');

        fill.style.width = '100%';
        setStatus('Готово. Можно вернуться в Telegram — бот начал обработку.', 'ok');
      }} catch (err) {{
        setStatus(err.message || String(err), 'err');
        btn.disabled = false;
      }}
    }});
  </script>
</body>
</html>
"""


def setup_upload_routes(
    app: web.Application,
    *,
    store: UploadSessionStore,
    storage: S3Storage,
    on_complete: UploadCompleteHandler,
) -> None:
    """Регистрирует страницу и API загрузки на существующем aiohttp app."""

    allowed_mimes = _allowed_mime_types()
    max_bytes = storage.config.max_upload_mb * 1024 * 1024

    async def page(request: web.Request) -> web.Response:
        upload_id = request.match_info["upload_id"]
        token = _extract_token(request)
        session = await store.get_authorized(upload_id, token)
        if session is None:
            return web.Response(
                text="Ссылка недействительна или устарела.",
                status=404,
                content_type="text/plain",
                charset="utf-8",
            )
        if session.status not in {"pending"}:
            return web.Response(
                text="Эта загрузка уже использована. Запросите новую ссылку в боте.",
                status=410,
                content_type="text/plain",
                charset="utf-8",
            )
        html = _page_html(upload_id, token, storage.config.max_upload_mb)
        return web.Response(text=html, content_type="text/html", charset="utf-8")

    async def presign(request: web.Request) -> web.Response:
        upload_id = request.match_info["upload_id"]
        token = _extract_token(request)
        session = await store.get_authorized(upload_id, token)
        if session is None:
            return web.json_response({"error": "Сессия не найдена"}, status=404)
        if session.status != "pending":
            return web.json_response({"error": "Сессия уже использована"}, status=409)

        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return web.json_response({"error": "Некорректный JSON"}, status=400)

        file_name = str(body.get("fileName") or "").strip()
        file_type = str(body.get("fileType") or "").strip().lower()
        file_size = body.get("fileSize")
        if not file_name:
            return web.json_response({"error": "fileName обязателен"}, status=400)
        lower_name = file_name.lower()
        if not any(lower_name.endswith(ext) for ext in ALLOWED_EXTENSIONS):
            return web.json_response({"error": "Недопустимое расширение файла"}, status=400)
        if file_type and file_type not in allowed_mimes and file_type != "application/octet-stream":
            return web.json_response(
                {"error": f"MIME не разрешён: {file_type}"}, status=400
            )
        if isinstance(file_size, int) and file_size > max_bytes:
            return web.json_response(
                {"error": f"Файл больше {storage.config.max_upload_mb} МБ"},
                status=400,
            )

        effective_type = file_type if file_type and file_type != "application/octet-stream" else _guess_mime(file_name)
        key = object_key_for(session, file_name)
        try:
            signed = await storage.create_presigned_put(
                object_key=key, content_type=effective_type
            )
        except S3StorageError as exc:
            return web.json_response({"error": exc.message}, status=500)

        session.object_key = key
        session.filename = file_name
        session.mime_type = effective_type
        if isinstance(file_size, int):
            session.file_size = file_size
        await store.update(session)

        return web.json_response(
            {
                "signedUrl": signed,
                "objectKey": key,
                "expiresIn": storage.config.signed_url_expires_in,
                "contentType": effective_type,
            }
        )

    async def complete(request: web.Request) -> web.Response:
        upload_id = request.match_info["upload_id"]
        token = _extract_token(request)
        session = await store.get_authorized(upload_id, token)
        if session is None:
            return web.json_response({"error": "Сессия не найдена"}, status=404)
        if session.status not in {"pending", "uploaded"}:
            return web.json_response({"error": "Сессия уже обработана"}, status=409)

        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return web.json_response({"error": "Некорректный JSON"}, status=400)

        object_key = str(body.get("objectKey") or session.object_key or "").strip()
        expected_prefix = f"uploads/{session.upload_id}/"
        if not object_key.startswith(expected_prefix):
            return web.json_response({"error": "objectKey не принадлежит сессии"}, status=400)

        try:
            head = await storage.head_object(object_key)
        except S3StorageError as exc:
            return web.json_response({"error": exc.message}, status=400)

        size = int(head.get("ContentLength") or 0)
        if size <= 0:
            return web.json_response({"error": "Пустой объект в бакете"}, status=400)
        if size > max_bytes:
            return web.json_response(
                {"error": f"Файл больше {storage.config.max_upload_mb} МБ"},
                status=400,
            )

        session.object_key = object_key
        session.filename = str(body.get("fileName") or session.filename or "media.bin")
        session.mime_type = str(
            body.get("fileType")
            or head.get("ContentType")
            or session.mime_type
            or "application/octet-stream"
        )
        session.file_size = size
        session.status = "uploaded"
        await store.update(session)

        # Сразу «processing», чтобы complete нельзя было дёрнуть дважды.
        session.status = "processing"
        await store.update(session)

        async def _run() -> None:
            try:
                await on_complete(session)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "upload_complete_handler_failed", upload_id=session.upload_id
                )
                session.status = "failed"
                session.error = "processing_failed"
                await store.update(session)

        asyncio.create_task(_run())
        return web.json_response({"ok": True, "status": "processing"})

    app.router.add_get("/upload/{upload_id}", page)
    app.router.add_post("/api/uploads/{upload_id}/presign", presign)
    app.router.add_post("/api/uploads/{upload_id}/complete", complete)
    logger.info("upload_routes_registered")


def _guess_mime(filename: str) -> str:
    lower = filename.lower()
    mapping = {
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".wav": "audio/wav",
        ".ogg": "audio/ogg",
        ".oga": "audio/ogg",
        ".webm": "video/webm",
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".mkv": "video/x-matroska",
    }
    for ext, mime in mapping.items():
        if lower.endswith(ext):
            return mime
    return "application/octet-stream"


def public_upload_url(upload_id: str, token: str) -> str | None:
    base = os.getenv("UPLOAD_PUBLIC_BASE_URL", "").strip().rstrip("/")
    if not base:
        return None
    return f"{base}/upload/{upload_id}?t={token}"


def is_upload_feature_ready() -> bool:
    from s3_storage import is_s3_configured

    return is_s3_configured() and bool(os.getenv("UPLOAD_PUBLIC_BASE_URL", "").strip())
