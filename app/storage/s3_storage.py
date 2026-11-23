"""Yandex Object Storage (S3-compatible) helpers for large media uploads."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger("s3_storage")


class S3StorageError(Exception):
    """Ошибка работы с объектным хранилищем."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class S3NotFoundError(S3StorageError):
    """Объекта нет в бакете. Остальные сбои S3 этим классом не маскируются."""


@dataclass(frozen=True, slots=True)
class S3Config:
    endpoint: str
    region: str
    bucket: str
    access_key_id: str
    secret_access_key: str
    signed_url_expires_in: int = 1800
    max_upload_mb: int = 512

    @classmethod
    def from_env(cls) -> S3Config | None:
        from app.infra.settings import get_settings

        settings = get_settings()
        bucket = settings.yc_bucket_name
        access_key_id = settings.yc_access_key_id
        secret_access_key = settings.yc_secret_access_key
        if not bucket or not access_key_id or not secret_access_key:
            return None
        from app.storage.s3_endpoint import s3_endpoint_url, valid_bucket_name

        if not valid_bucket_name(bucket):
            logger.error("s3_bucket_name_rejected", bucket=bucket)
            raise SystemExit(1)
        endpoint = s3_endpoint_url(settings.yc_s3_endpoint)
        return cls(
            endpoint=endpoint,
            region=settings.yc_region,
            bucket=bucket,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            signed_url_expires_in=settings.signed_url_expires_in,
            max_upload_mb=settings.upload_max_mb,
        )


_MISSING_CODES = frozenset({"404", "NoSuchKey", "NotFound"})


def _storage_error_types() -> tuple[type[BaseException], ...]:
    """Ошибки SDK и локального чтения файла. Сбои кода сюда не входят.

    ``ClientError`` (ответ бакета: 403, 404, 500) не наследник ``BotoCoreError``.
    """
    try:
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError:  # pragma: no cover - dependency guard
        return (OSError,)
    return (BotoCoreError, ClientError, OSError)


def _error_code(exc: BaseException) -> str:
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        error = response.get("Error")
        if isinstance(error, dict) and error.get("Code"):
            return str(error["Code"])
    return type(exc).__name__


def _is_missing(exc: BaseException) -> bool:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    error = response.get("Error")
    code = str(error.get("Code") or "") if isinstance(error, dict) else ""
    meta = response.get("ResponseMetadata")
    status = meta.get("HTTPStatusCode") if isinstance(meta, dict) else None
    return code in _MISSING_CODES or status == 404


def _build_client(config: S3Config) -> Any:
    try:
        import boto3
        from botocore.config import Config as BotoConfig
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise S3StorageError(
            "Пакет boto3 не установлен. Добавьте boto3 в requirements."
        ) from exc

    from urllib.parse import urlsplit

    from app.storage.s3_endpoint import (
        YANDEX_S3_HOST,
        allowed_s3_hosts,
        guard_s3_request,
    )

    # virtual-hosted: https://{bucket}.storage.yandexcloud.net/... — так же, как
    # у рабочего MRISchoolClub / AWS SDK; path-style чаще ломает CORS в браузере.
    # На loopback path-style, иначе хост становится {bucket}.localhost.
    host = urlsplit(config.endpoint).hostname
    style = "virtual" if host == YANDEX_S3_HOST else "path"
    client = boto3.client(
        "s3",
        endpoint_url=config.endpoint,
        region_name=config.region,
        aws_access_key_id=config.access_key_id,
        aws_secret_access_key=config.secret_access_key,
        config=BotoConfig(
            signature_version="s3v4",
            s3={"addressing_style": style},
        ),
    )
    allowed = allowed_s3_hosts(config.endpoint, config.bucket)

    def _guard(request, **_kwargs: object) -> None:
        guard_s3_request(getattr(request, "url", ""), allowed=allowed)

    client.meta.events.register("before-send.s3", _guard)
    return client


class S3Storage:
    """Синхронный boto3 под asyncio.to_thread."""

    def __init__(self, config: S3Config) -> None:
        self.config = config
        self._client = _build_client(config)

    async def create_presigned_put(self, *, object_key: str, content_type: str) -> str:
        def _sign() -> str:
            from app.storage.s3_endpoint import allowed_s3_hosts, guard_s3_request

            signed = self._client.generate_presigned_url(
                "put_object",
                Params={
                    "Bucket": self.config.bucket,
                    "Key": object_key,
                    "ContentType": content_type,
                },
                ExpiresIn=self.config.signed_url_expires_in,
                HttpMethod="PUT",
            )
            guard_s3_request(
                signed,
                allowed=allowed_s3_hosts(self.config.endpoint, self.config.bucket),
            )
            return signed

        try:
            return await asyncio.to_thread(_sign)
        except _storage_error_types() as exc:
            logger.exception(
                "s3_presign_failed", key=object_key, error=_error_code(exc)
            )
            raise S3StorageError("Не удалось создать ссылку загрузки") from exc

    async def upload_file(
        self, *, object_key: str, file_path: str, content_type: str
    ) -> None:
        """Загружает локальный файл в бакет (серверный PUT, без CORS браузера)."""

        def _upload() -> None:
            self._client.upload_file(
                file_path,
                self.config.bucket,
                object_key,
                ExtraArgs={"ContentType": content_type},
            )

        try:
            await asyncio.to_thread(_upload)
        except _storage_error_types() as exc:
            logger.exception(
                "s3_upload_file_failed", key=object_key, error=_error_code(exc)
            )
            raise S3StorageError("Не удалось загрузить объект") from exc

    async def head_object(self, object_key: str) -> dict[str, Any]:
        def _head() -> dict[str, Any]:
            return self._client.head_object(Bucket=self.config.bucket, Key=object_key)

        try:
            return await asyncio.to_thread(_head)
        except _storage_error_types() as exc:
            if _is_missing(exc):
                raise S3NotFoundError("Объект не найден в бакете") from exc
            logger.exception("s3_head_failed", key=object_key, error=_error_code(exc))
            raise S3StorageError("Не удалось проверить объект в бакете") from exc

    async def download_bytes(
        self, object_key: str, *, max_bytes: int | None = None
    ) -> bytes:
        """Download object body, optionally refusing payloads above ``max_bytes``."""
        limit = max_bytes
        if limit is None:
            limit = self.config.max_upload_mb * 1024 * 1024

        def _get() -> bytes:
            response = self._client.get_object(
                Bucket=self.config.bucket, Key=object_key
            )
            content_length = response.get("ContentLength")
            body = response["Body"]
            try:
                if content_length is not None and int(content_length) > limit:
                    raise S3StorageError(
                        f"Объект больше лимита ({limit} байт): {object_key}"
                    )
                chunks: list[bytes] = []
                total = 0
                while True:
                    chunk = body.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > limit:
                        raise S3StorageError(
                            f"Объект больше лимита ({limit} байт): {object_key}"
                        )
                    chunks.append(chunk)
                return b"".join(chunks)
            finally:
                body.close()

        try:
            return await asyncio.to_thread(_get)
        except S3StorageError:
            raise
        except _storage_error_types() as exc:
            logger.exception(
                "s3_download_failed", key=object_key, error=_error_code(exc)
            )
            raise S3StorageError("Не удалось скачать объект") from exc

    async def delete_object(self, object_key: str) -> None:
        def _delete() -> None:
            self._client.delete_object(Bucket=self.config.bucket, Key=object_key)

        try:
            await asyncio.to_thread(_delete)
        except _storage_error_types() as exc:
            logger.warning("s3_delete_failed", key=object_key, error=_error_code(exc))


def is_s3_configured() -> bool:
    return S3Config.from_env() is not None


def get_storage() -> S3Storage | None:
    config = S3Config.from_env()
    if config is None:
        return None
    return S3Storage(config)
