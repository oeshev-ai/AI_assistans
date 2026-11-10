"""Сессии загрузки больших медиа (Redis или память процесса)."""

from __future__ import annotations

import json
import secrets
import time
import uuid
from contextvars import ContextVar, Token
from dataclasses import asdict, dataclass
from typing import Literal

import structlog
from redis.asyncio import Redis

logger = structlog.get_logger("upload_sessions")

UploadPurpose = Literal["transcribe"]
UploadMediaKind = Literal["audio", "video", "auto"]
UploadStatus = Literal["pending", "uploaded", "processing", "done", "failed"]

_SESSION_TTL_SEC = 60 * 60  # 1 час
_REDIS_PREFIX = "upload:session:"


@dataclass(slots=True)
class UploadRequestContext:
    chat_id: int
    user_id: int


@dataclass(slots=True)
class UploadSession:
    upload_id: str
    token: str
    chat_id: int
    user_id: int
    purpose: UploadPurpose
    media_kind: UploadMediaKind
    status: UploadStatus
    created_at: float
    expires_at: float
    object_key: str | None = None
    filename: str | None = None
    mime_type: str | None = None
    file_size: int | None = None
    error: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> UploadSession:
        data = json.loads(raw)
        return cls(**data)


_REQUEST_CTX: ContextVar[UploadRequestContext | None] = ContextVar(
    "upload_request_context", default=None
)


def set_upload_request_context(
    ctx: UploadRequestContext,
) -> Token[UploadRequestContext | None]:
    return _REQUEST_CTX.set(ctx)


def reset_upload_request_context(
    token: Token[UploadRequestContext | None],
) -> None:
    _REQUEST_CTX.reset(token)


def get_upload_request_context() -> UploadRequestContext | None:
    return _REQUEST_CTX.get()


class UploadSessionStore:
    """Хранит одноразовые сессии загрузки."""

    def __init__(self, redis: Redis | None = None) -> None:
        self._redis = redis
        self._memory: dict[str, UploadSession] = {}

    async def create(
        self,
        *,
        chat_id: int,
        user_id: int,
        purpose: UploadPurpose = "transcribe",
        media_kind: UploadMediaKind = "auto",
        ttl_sec: int = _SESSION_TTL_SEC,
    ) -> UploadSession:
        now = time.time()
        session = UploadSession(
            upload_id=uuid.uuid4().hex,
            token=secrets.token_urlsafe(24),
            chat_id=chat_id,
            user_id=user_id,
            purpose=purpose,
            media_kind=media_kind,
            status="pending",
            created_at=now,
            expires_at=now + ttl_sec,
        )
        await self._save(session, ttl_sec=ttl_sec)
        logger.info(
            "upload_session_created",
            upload_id=session.upload_id,
            chat_id=chat_id,
            user_id=user_id,
            purpose=purpose,
        )
        return session

    async def get(self, upload_id: str) -> UploadSession | None:
        if self._redis is not None:
            raw = await self._redis.get(f"{_REDIS_PREFIX}{upload_id}")
            if not raw:
                return None
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            session = UploadSession.from_json(raw)
        else:
            session = self._memory.get(upload_id)
            if session is None:
                return None
        if session.expires_at < time.time():
            await self.delete(upload_id)
            return None
        return session

    async def get_authorized(
        self, upload_id: str, token: str
    ) -> UploadSession | None:
        session = await self.get(upload_id)
        if session is None or not secrets.compare_digest(session.token, token):
            return None
        return session

    async def update(self, session: UploadSession) -> None:
        ttl = max(60, int(session.expires_at - time.time()))
        await self._save(session, ttl_sec=ttl)

    async def delete(self, upload_id: str) -> None:
        if self._redis is not None:
            await self._redis.delete(f"{_REDIS_PREFIX}{upload_id}")
        self._memory.pop(upload_id, None)

    async def _save(self, session: UploadSession, *, ttl_sec: int) -> None:
        if self._redis is not None:
            await self._redis.set(
                f"{_REDIS_PREFIX}{session.upload_id}",
                session.to_json(),
                ex=max(60, ttl_sec),
            )
        else:
            self._memory[session.upload_id] = session


def object_key_for(session: UploadSession, filename: str) -> str:
    safe = _sanitize_filename(filename)
    return f"uploads/{session.upload_id}/{int(time.time())}-{safe}"


def _sanitize_filename(file_name: str) -> str:
    base = file_name.replace("\\", "/").split("/")[-1].strip() or "file.bin"
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in base)
    return cleaned[:180] or "file.bin"
