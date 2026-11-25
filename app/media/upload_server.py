"""HTTP: страница загрузки + presign/complete для Yandex Object Storage."""

from __future__ import annotations

import asyncio
import json
import re
import secrets
import tempfile
import time
from collections.abc import Awaitable, Callable
from http.cookies import SimpleCookie
from pathlib import Path
from typing import Any, Protocol

import structlog
from aiohttp import BodyPartReader, web

from app.storage.s3_storage import S3NotFoundError, S3StorageError
from app.storage.upload_sessions import (
    UploadSession,
    UploadSessionStore,
    object_key_for,
)

logger = structlog.get_logger("upload_server")

UploadCompleteHandler = Callable[[UploadSession], Awaitable[None]]

_COOKIE_NAME = "upload_token"
_UPLOAD_ID_RE = re.compile(r"^[0-9a-f]{32}$")
# Страница и API. path="/" уносил бы токен на /live и /ready того же порта.
_COOKIE_PATHS = ("/upload/", "/api/uploads/")
UPLOAD_TASKS_KEY: web.AppKey[set[asyncio.Task[Any]]] = web.AppKey(
    "upload_tasks", set[asyncio.Task[Any]]
)


class UploadObjectStorage(Protocol):
    """Минимальный контракт хранилища (S3Storage или тест-двойник)."""

    config: Any

    async def create_presigned_put(
        self, *, object_key: str, content_type: str
    ) -> str: ...

    async def upload_file(
        self, *, object_key: str, file_path: str, content_type: str
    ) -> None: ...

    async def head_object(self, object_key: str) -> dict[str, Any]: ...


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

# Запас на boundary и заголовки multipart, чтобы Content-Length чуть выше
# файла на лимите не отвергался заранее.
_MULTIPART_FRAMING_BYTES = 64 * 1024
_UPLOAD_CHUNK = 1024 * 1024


async def _copy_until_limit(field: Any, dest: Any, max_bytes: int) -> tuple[int, bool]:
    """Копирует поле, пока оно в лимите. Хвост после превышения не читает."""
    size = 0
    while True:
        chunk = await field.read_chunk(size=_UPLOAD_CHUNK)
        if not chunk:
            return size, False
        size += len(chunk)
        if size > max_bytes:
            return size, True
        await asyncio.to_thread(dest.write, chunk)


def _payload_too_large(request: web.Request, limit_mb: int) -> web.Response:
    """413 и закрытие сокета. Иначе aiohttp дочитывает тело lingering_time секунд.

    ``_lingering_time`` — приватное поле aiohttp 3.x. Если его не станет,
    присваивание создало бы новый атрибут и linger вернулся бы без ошибки.
    """
    protocol = request.protocol
    if protocol is not None and hasattr(protocol, "_lingering_time"):
        try:
            protocol._lingering_time = 0.0
        except Exception as exc:
            logger.warning("upload_linger_disable_failed", error=str(exc))
    response = web.json_response(
        {"error": f"Файл больше {limit_mb} МБ"},
        status=413,
        headers={"Connection": "close"},
    )
    response.force_close()
    return response


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


def _declared_file_size(value: object) -> int | None:
    """Размер из JSON. ``true``/``false`` не размер: ``bool`` — подкласс ``int``."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _allowed_mime_types() -> set[str]:
    from app.infra.settings import get_settings

    raw = get_settings().upload_allowed_mime_types
    if not raw:
        return set(DEFAULT_ALLOWED_MIME)
    return {item.strip() for item in raw.split(",") if item.strip()}


def _trust_proxy() -> bool:
    from app.infra.settings import get_settings

    return get_settings().trust_proxy


def _legacy_presign_enabled() -> bool:
    from app.infra.settings import get_settings

    return get_settings().upload_legacy_presign


def _request_is_https(request: web.Request) -> bool:
    """HTTPS for Secure cookies. Trust X-Forwarded-Proto only if TRUST_PROXY=1."""
    if request.secure:
        return True
    if not _trust_proxy():
        return False
    return request.headers.get("X-Forwarded-Proto", "").strip().lower() == "https"


def _client_key(request: web.Request) -> str:
    """IP для лимита. За доверенным прокси берётся адрес, который дописал nginx."""
    if _trust_proxy():
        forwarded = request.headers.get("X-Forwarded-For", "")
        parts = [part.strip() for part in forwarded.split(",") if part.strip()]
        if parts:
            return parts[-1][:128]
    return (request.remote or "unknown")[:128]


async def _upload_rate_limited(
    request: web.Request, *, write: bool
) -> web.Response | None:
    """429 до чтения тела. ``None`` — слот есть, обработчик продолжается."""
    from app.infra.rate_limiter import rate_limiter

    identity = _client_key(request)
    if await rate_limiter.check_upload_limit(identity, write=write):
        return None
    retry = max(
        1, int(await rate_limiter.get_upload_retry_after(identity, write=write))
    )
    logger.warning("upload_rate_limited", client=identity, write=write)
    headers = {"Retry-After": str(retry)}
    if write:
        return web.json_response(
            {"error": "Слишком много запросов. Попробуйте позже."},
            status=429,
            headers=headers,
        )
    return web.Response(
        text="Слишком много запросов. Попробуйте позже.",
        status=429,
        content_type="text/plain",
        charset="utf-8",
        headers=headers,
    )


def _extract_token(request: web.Request) -> str:
    # Prefer cookie / headers so the token does not linger in URLs after redirect.
    cookie = (request.cookies.get(_COOKIE_NAME) or "").strip()
    if cookie:
        return cookie
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    header = (request.headers.get("X-Upload-Token") or "").strip()
    if header:
        return header
    return request.rel_url.query.get("t", "").strip()


def _set_upload_cookie(
    response: web.Response,
    token: str,
    *,
    max_age: int,
    secure: bool,
) -> None:
    """Два Set-Cookie с одним именем: SimpleCookie хранит только один path."""
    age = str(max(60, max_age))
    for path in _COOKIE_PATHS:
        baked = SimpleCookie()
        baked[_COOKIE_NAME] = token
        morsel = baked[_COOKIE_NAME]
        morsel["max-age"] = age
        morsel["httponly"] = True
        morsel["samesite"] = "Lax"
        morsel["path"] = path
        if secure:
            morsel["secure"] = True
        response.headers.add("Set-Cookie", morsel.OutputString())


def _bad_upload_id(upload_id: str, *, page: bool) -> web.Response | None:
    """None — id годится. Иначе 404 до стора и до HTML."""
    if _UPLOAD_ID_RE.fullmatch(upload_id):
        return None
    if page:
        return web.Response(
            text="Ссылка недействительна или устарела.",
            status=404,
            content_type="text/plain",
            charset="utf-8",
        )
    return web.json_response({"error": "Сессия не найдена"}, status=404)


def _page_html(upload_id: str, max_mb: int) -> str:
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
    <input id="file" type="file" accept="{",".join(ALLOWED_EXTENSIONS)},audio/*,video/*"/>
    <button id="btn" type="button">Загрузить</button>
    <div class="bar" id="bar"><span id="fill"></span></div>
    <div class="status" id="status"></div>
  </div>
  <script>
    const UPLOAD_ID = {json.dumps(upload_id)};
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
        // Файл идёт на наш домен (same-origin), сервер сам кладёт в S3 —
        // так не нужен браузерный CORS к storage.yandexcloud.net (Telegram WebView).
        // Auth: HttpOnly cookie set by /upload redirect (no token in JS/query).
        setStatus('Загружаю файл…');
        const form = new FormData();
        form.append('file', file, file.name);
        const uploadUrl = '/api/uploads/' + UPLOAD_ID + '/file';

        await new Promise((resolve, reject) => {{
          const xhr = new XMLHttpRequest();
          xhr.open('POST', uploadUrl);
          xhr.withCredentials = true;
          xhr.timeout = 60 * 60 * 1000;
          xhr.upload.onprogress = (e) => {{
            if (e.lengthComputable) fill.style.width = Math.round(e.loaded / e.total * 100) + '%';
          }};
          xhr.onload = () => {{
            let payload = null;
            try {{ payload = JSON.parse(xhr.responseText || '{{}}'); }} catch (_e) {{}}
            if (xhr.status >= 200 && xhr.status < 300) resolve(payload);
            else if (xhr.status === 413) {{
              reject(new Error((payload && payload.error) || 'Файл слишком большой'));
            }} else {{
              reject(new Error((payload && payload.error) || ('Ошибка загрузки (' + xhr.status + ')')));
            }}
          }};
          xhr.ontimeout = () => reject(new Error('Таймаут загрузки. Проверьте proxy_read_timeout в nginx.'));
          xhr.onerror = () => reject(new Error(
            'Обрыв соединения с сервером.'
          ));
          xhr.send(form);
        }});

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


def track_upload_task(app: web.Application, task: asyncio.Task[Any]) -> None:
    """Keep background upload jobs so shutdown can cancel/await them."""
    tasks: set[asyncio.Task[Any]] = app.setdefault(UPLOAD_TASKS_KEY, set())
    tasks.add(task)
    task.add_done_callback(tasks.discard)


async def drain_upload_tasks(app: web.Application, *, timeout: float = 30.0) -> None:
    """Cancel in-flight upload completion tasks during process shutdown."""
    tasks: set[asyncio.Task[Any]] = set(app.get(UPLOAD_TASKS_KEY) or ())
    if not tasks:
        return
    logger.info("upload_tasks_draining", count=len(tasks))
    for task in tasks:
        task.cancel()
    _done, pending = await asyncio.wait(tasks, timeout=timeout)
    if pending:
        logger.warning("upload_tasks_drain_timeout", pending=len(pending))


def setup_upload_routes(
    app: web.Application,
    *,
    store: UploadSessionStore,
    storage: UploadObjectStorage,
    on_complete: UploadCompleteHandler,
) -> None:
    """Регистрирует страницу и API загрузки на существующем aiohttp app."""

    app.setdefault(UPLOAD_TASKS_KEY, set())
    allowed_mimes = _allowed_mime_types()
    max_bytes = storage.config.max_upload_mb * 1024 * 1024

    async def page(request: web.Request) -> web.Response:
        limited = await _upload_rate_limited(request, write=False)
        if limited is not None:
            return limited
        upload_id = request.match_info["upload_id"]
        bad = _bad_upload_id(upload_id, page=True)
        if bad is not None:
            return bad
        query_token = request.rel_url.query.get("t", "").strip()
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
        # First hop from Telegram still has ?t= — exchange for HttpOnly cookie
        # and drop the token from the address bar / subsequent Referer.
        if query_token and secrets.compare_digest(query_token, session.token):
            max_age = max(60, int(session.expires_at - time.time()))
            redirect = web.HTTPFound(f"/upload/{upload_id}")
            _set_upload_cookie(
                redirect,
                session.token,
                max_age=max_age,
                secure=_request_is_https(request),
            )
            raise redirect
        html = _page_html(upload_id, storage.config.max_upload_mb)
        response = web.Response(text=html, content_type="text/html", charset="utf-8")
        if not request.cookies.get(_COOKIE_NAME):
            max_age = max(60, int(session.expires_at - time.time()))
            _set_upload_cookie(
                response,
                session.token,
                max_age=max_age,
                secure=_request_is_https(request),
            )
        return response

    async def presign(request: web.Request) -> web.Response:
        limited = await _upload_rate_limited(request, write=True)
        if limited is not None:
            return limited
        upload_id = request.match_info["upload_id"]
        bad = _bad_upload_id(upload_id, page=False)
        if bad is not None:
            return bad
        token = _extract_token(request)
        session = await store.get_authorized(upload_id, token)
        if session is None:
            return web.json_response({"error": "Сессия не найдена"}, status=404)
        if session.status != "pending":
            return web.json_response({"error": "Сессия уже использована"}, status=409)

        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Некорректный JSON"}, status=400)

        file_name = str(body.get("fileName") or "").strip()
        file_type = str(body.get("fileType") or "").strip().lower()
        file_size = _declared_file_size(body.get("fileSize"))
        if not file_name:
            return web.json_response({"error": "fileName обязателен"}, status=400)
        lower_name = file_name.lower()
        if not any(lower_name.endswith(ext) for ext in ALLOWED_EXTENSIONS):
            return web.json_response(
                {"error": "Недопустимое расширение файла"}, status=400
            )
        if (
            file_type
            and file_type not in allowed_mimes
            and file_type != "application/octet-stream"
        ):
            return web.json_response(
                {"error": f"MIME не разрешён: {file_type}"}, status=400
            )
        if file_size is not None and file_size > max_bytes:
            return web.json_response(
                {"error": f"Файл больше {storage.config.max_upload_mb} МБ"},
                status=400,
            )

        effective_type = (
            file_type
            if file_type and file_type != "application/octet-stream"
            else _guess_mime(file_name)
        )
        key = object_key_for(session, file_name)
        try:
            signed = await storage.create_presigned_put(
                object_key=key, content_type=effective_type
            )
        except S3StorageError as exc:
            logger.warning("upload_presign_s3_failed", error=str(exc.message))
            return web.json_response(
                {"error": "Ошибка подготовки загрузки"}, status=500
            )

        session.object_key = key
        session.filename = file_name
        session.mime_type = effective_type
        if file_size is not None:
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
        limited = await _upload_rate_limited(request, write=True)
        if limited is not None:
            return limited
        upload_id = request.match_info["upload_id"]
        bad = _bad_upload_id(upload_id, page=False)
        if bad is not None:
            return bad
        token = _extract_token(request)
        session = await store.get_authorized(upload_id, token)
        if session is None:
            return web.json_response({"error": "Сессия не найдена"}, status=404)
        if session.status not in {"pending", "uploaded"}:
            return web.json_response({"error": "Сессия уже обработана"}, status=409)

        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Некорректный JSON"}, status=400)
        if not isinstance(body, dict):
            return web.json_response({"error": "Некорректный JSON"}, status=400)

        object_key = str(body.get("objectKey") or session.object_key or "").strip()
        expected_prefix = f"uploads/{session.upload_id}/"
        if not object_key.startswith(expected_prefix):
            return web.json_response(
                {"error": "objectKey не принадлежит сессии"}, status=400
            )

        try:
            head = await storage.head_object(object_key)
        except S3NotFoundError:
            return web.json_response({"error": "Объект не найден"}, status=400)
        except S3StorageError as exc:
            logger.warning("upload_complete_head_failed", error=str(exc.message))
            return web.json_response({"error": "Ошибка проверки загрузки"}, status=500)

        size = int(head.get("ContentLength") or 0)
        if size <= 0:
            return web.json_response({"error": "Пустой объект в бакете"}, status=400)
        if size > max_bytes:
            return web.json_response(
                {"error": f"Файл больше {storage.config.max_upload_mb} МБ"},
                status=400,
            )

        claimed = await store.claim_processing(
            upload_id,
            token,
            from_statuses={"pending", "uploaded"},
            updates={
                "object_key": object_key,
                "filename": str(
                    body.get("fileName") or session.filename or "media.bin"
                ),
                "mime_type": str(
                    body.get("fileType")
                    or head.get("ContentType")
                    or session.mime_type
                    or "application/octet-stream"
                ),
                "file_size": size,
            },
        )
        if claimed is None:
            return web.json_response({"error": "Сессия уже обработана"}, status=409)

        async def _run() -> None:
            try:
                await on_complete(claimed)
            except Exception:
                logger.exception(
                    "upload_complete_handler_failed", upload_id=claimed.upload_id
                )
                claimed.status = "failed"
                claimed.error = "processing_failed"
                await store.update(claimed)

        track_upload_task(app, asyncio.create_task(_run()))
        return web.json_response({"ok": True, "status": "processing"})

    async def upload_file(request: web.Request) -> web.Response:
        """Same-origin multipart → S3 (обход CORS / Telegram WebView)."""
        limited = await _upload_rate_limited(request, write=True)
        if limited is not None:
            return limited
        upload_id = request.match_info["upload_id"]
        bad = _bad_upload_id(upload_id, page=False)
        if bad is not None:
            return bad
        token = _extract_token(request)
        # Claim early so two parallel POSTs cannot both process.
        session = await store.claim_processing(
            upload_id, token, from_statuses={"pending"}
        )
        if session is None:
            existing = await store.get_authorized(upload_id, token)
            if existing is None:
                return web.json_response({"error": "Сессия не найдена"}, status=404)
            return web.json_response({"error": "Сессия уже использована"}, status=409)

        async def _release_claim(error: str | None = None) -> None:
            session.status = "pending" if error is None else "failed"
            session.error = error
            await store.update(session)

        # Весь разбор multipart внутри try: иначе claim остаётся в processing.
        tmp_path: str | None = None
        size = 0
        too_large = False
        try:
            declared = request.content_length
            if declared is not None and declared > max_bytes + _MULTIPART_FRAMING_BYTES:
                await _release_claim()
                return _payload_too_large(request, storage.config.max_upload_mb)
            try:
                reader = await request.multipart()
                field = await reader.next()
            except Exception:
                logger.exception("upload_file_multipart_failed", upload_id=upload_id)
                # pending — клиент может повторить; processing не должен «зависнуть».
                await _release_claim()
                return web.json_response(
                    {"error": "Некорректное тело запроса"}, status=400
                )
            if not isinstance(field, BodyPartReader) or field.name != "file":
                await _release_claim()
                return web.json_response({"error": "Ожидалось поле file"}, status=400)

            file_name = Path(field.filename or "media.bin").name
            lower_name = file_name.lower()
            if not any(lower_name.endswith(ext) for ext in ALLOWED_EXTENSIONS):
                await _release_claim()
                return web.json_response(
                    {"error": "Недопустимое расширение файла"}, status=400
                )

            header_ctype = (
                (field.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            )
            effective_type = header_ctype or _guess_mime(file_name)
            if effective_type == "application/octet-stream":
                effective_type = _guess_mime(file_name)
            if (
                effective_type not in allowed_mimes
                and effective_type != "application/octet-stream"
            ):
                await _release_claim()
                return web.json_response(
                    {"error": f"MIME не разрешён: {effective_type}"}, status=400
                )

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(file_name).suffix
            ) as tmp:
                tmp_path = tmp.name
                size, too_large = await _copy_until_limit(field, tmp, max_bytes)

            if too_large:
                await _release_claim()
                return _payload_too_large(request, storage.config.max_upload_mb)
            if size <= 0:
                await _release_claim()
                return web.json_response({"error": "Пустой файл"}, status=400)

            from app.media.file_magic import mime_compatible

            if not await mime_compatible(
                filename=file_name,
                claimed_mime=effective_type,
                source=Path(tmp_path),
            ):
                await _release_claim()
                return web.json_response(
                    {"error": "Содержимое файла не соответствует типу"},
                    status=400,
                )

            key = object_key_for(session, file_name)
            await storage.upload_file(
                object_key=key,
                file_path=tmp_path,
                content_type=effective_type,
            )

            session.object_key = key
            session.filename = file_name
            session.mime_type = effective_type
            session.verified_mime = effective_type
            session.file_size = size
            session.status = "processing"
            await store.update(session)

            async def _run() -> None:
                try:
                    await on_complete(session)
                except Exception:
                    logger.exception(
                        "upload_file_handler_failed", upload_id=session.upload_id
                    )
                    session.status = "failed"
                    session.error = "processing_failed"
                    await store.update(session)

            track_upload_task(app, asyncio.create_task(_run()))
            return web.json_response(
                {
                    "ok": True,
                    "status": "processing",
                    "objectKey": key,
                    "fileSize": size,
                }
            )
        except S3StorageError as exc:
            logger.warning(
                "upload_file_s3_failed",
                upload_id=upload_id,
                error=str(exc.message),
            )
            await _release_claim(error="s3_upload_failed")
            return web.json_response({"error": "Ошибка загрузки"}, status=500)
        except Exception:
            logger.exception("upload_file_failed", upload_id=upload_id)
            await _release_claim(error="upload_failed")
            return web.json_response({"error": "Ошибка загрузки"}, status=500)
        finally:
            if tmp_path:
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except Exception:
                    logger.debug("upload_temp_cleanup_failed", path=tmp_path)

    app.router.add_get("/upload/{upload_id}", page)
    app.router.add_post("/api/uploads/{upload_id}/file", upload_file)
    if _legacy_presign_enabled():
        app.router.add_post("/api/uploads/{upload_id}/presign", presign)
        app.router.add_post("/api/uploads/{upload_id}/complete", complete)
        logger.info("upload_legacy_presign_enabled")
    logger.info("upload_routes_registered", legacy_presign=_legacy_presign_enabled())


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
    from app.infra.settings import get_settings

    base = get_settings().upload_public_base_url.rstrip("/")
    if not base:
        return None
    return f"{base}/upload/{upload_id}?t={token}"


def is_upload_feature_ready() -> bool:
    from app.infra.settings import get_settings
    from app.storage.s3_storage import is_s3_configured

    return is_s3_configured() and bool(get_settings().upload_public_base_url)
