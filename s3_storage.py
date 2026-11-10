"""Yandex Object Storage (S3-compatible) helpers for large media uploads."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger("s3_storage")


class S3StorageError(Exception):
    """Ошибка работы с объектным хранилищем."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


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
        bucket = os.getenv("YC_BUCKET_NAME", "").strip()
        access_key_id = os.getenv("YC_ACCESS_KEY_ID", "").strip()
        secret_access_key = os.getenv("YC_SECRET_ACCESS_KEY", "").strip()
        if not bucket or not access_key_id or not secret_access_key:
            return None
        return cls(
            endpoint=os.getenv(
                "YC_S3_ENDPOINT", "https://storage.yandexcloud.net"
            ).strip()
            or "https://storage.yandexcloud.net",
            region=os.getenv("YC_REGION", "ru-central1").strip() or "ru-central1",
            bucket=bucket,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            signed_url_expires_in=max(
                60, int(os.getenv("SIGNED_URL_EXPIRES_IN", "1800") or "1800")
            ),
            max_upload_mb=max(1, int(os.getenv("UPLOAD_MAX_MB", "512") or "512")),
        )


def _build_client(config: S3Config) -> Any:
    try:
        import boto3
        from botocore.config import Config as BotoConfig
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise S3StorageError(
            "Пакет boto3 не установлен. Добавьте boto3 в requirements."
        ) from exc

    # virtual-hosted: https://{bucket}.storage.yandexcloud.net/... — так же, как
    # у рабочего MRISchoolClub / AWS SDK; path-style чаще ломает CORS в браузере.
    return boto3.client(
        "s3",
        endpoint_url=config.endpoint,
        region_name=config.region,
        aws_access_key_id=config.access_key_id,
        aws_secret_access_key=config.secret_access_key,
        config=BotoConfig(
            signature_version="s3v4",
            s3={"addressing_style": "virtual"},
        ),
    )


class S3Storage:
    """Синхронный boto3 под asyncio.to_thread."""

    def __init__(self, config: S3Config) -> None:
        self.config = config
        self._client = _build_client(config)

    async def create_presigned_put(
        self, *, object_key: str, content_type: str
    ) -> str:
        def _sign() -> str:
            return self._client.generate_presigned_url(
                "put_object",
                Params={
                    "Bucket": self.config.bucket,
                    "Key": object_key,
                    "ContentType": content_type,
                },
                ExpiresIn=self.config.signed_url_expires_in,
                HttpMethod="PUT",
            )

        try:
            return await asyncio.to_thread(_sign)
        except Exception as exc:  # noqa: BLE001
            logger.exception("s3_presign_failed", key=object_key)
            raise S3StorageError(f"Не удалось создать ссылку загрузки: {exc}") from exc

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
        except Exception as exc:  # noqa: BLE001
            logger.exception("s3_upload_file_failed", key=object_key)
            raise S3StorageError(f"Не удалось загрузить объект: {exc}") from exc

    async def head_object(self, object_key: str) -> dict[str, Any]:
        def _head() -> dict[str, Any]:
            return self._client.head_object(
                Bucket=self.config.bucket, Key=object_key
            )

        try:
            return await asyncio.to_thread(_head)
        except Exception as exc:  # noqa: BLE001
            raise S3StorageError(f"Объект не найден в бакете: {object_key}") from exc

    async def download_bytes(self, object_key: str) -> bytes:
        def _get() -> bytes:
            response = self._client.get_object(
                Bucket=self.config.bucket, Key=object_key
            )
            body = response["Body"]
            try:
                return body.read()
            finally:
                body.close()

        try:
            return await asyncio.to_thread(_get)
        except Exception as exc:  # noqa: BLE001
            logger.exception("s3_download_failed", key=object_key)
            raise S3StorageError(f"Не удалось скачать объект: {exc}") from exc

    async def delete_object(self, object_key: str) -> None:
        def _delete() -> None:
            self._client.delete_object(Bucket=self.config.bucket, Key=object_key)

        try:
            await asyncio.to_thread(_delete)
        except Exception as exc:  # noqa: BLE001
            logger.warning("s3_delete_failed", key=object_key, error=str(exc))


def is_s3_configured() -> bool:
    return S3Config.from_env() is not None


def get_storage() -> S3Storage | None:
    config = S3Config.from_env()
    if config is None:
        return None
    return S3Storage(config)
