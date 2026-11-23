"""S3-слой ловит ошибки хранилища и не маскирует сбои кода."""

from __future__ import annotations

from typing import Any

import pytest

from app.storage.s3_storage import S3Config, S3NotFoundError, S3Storage, S3StorageError


def _storage(client: Any) -> S3Storage:
    storage = S3Storage.__new__(S3Storage)
    storage.config = S3Config(
        endpoint="https://storage.yandexcloud.net",
        region="ru-central1",
        bucket="bucket",
        access_key_id="key",
        secret_access_key="secret",
    )
    storage._client = client
    return storage


def _client_error(code: str, *, status: int, operation: str) -> Exception:
    from botocore.exceptions import ClientError

    return ClientError(
        {
            "Error": {"Code": code, "Message": "https://secret.example/internal"},
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        operation,
    )


class _Boom:
    def __init__(self, error: BaseException) -> None:
        self._error = error

    def generate_presigned_url(self, *_args: Any, **_kwargs: Any) -> str:
        raise self._error

    def upload_file(self, *_args: Any, **_kwargs: Any) -> None:
        raise self._error

    def head_object(self, **_kwargs: Any) -> dict[str, Any]:
        raise self._error

    def delete_object(self, **_kwargs: Any) -> None:
        raise self._error


@pytest.mark.asyncio
async def test_presign_wraps_s3_error_without_its_text() -> None:
    storage = _storage(
        _Boom(_client_error("AccessDenied", status=403, operation="PutObject"))
    )
    with pytest.raises(S3StorageError) as caught:
        await storage.create_presigned_put(
            object_key="uploads/a", content_type="audio/wav"
        )
    assert "secret.example" not in str(caught.value)
    assert not isinstance(caught.value, S3NotFoundError)


@pytest.mark.asyncio
async def test_presign_lets_programming_errors_through() -> None:
    storage = _storage(_Boom(AttributeError("bug")))
    with pytest.raises(AttributeError):
        await storage.create_presigned_put(
            object_key="uploads/a", content_type="audio/wav"
        )


@pytest.mark.asyncio
async def test_head_missing_is_not_found() -> None:
    storage = _storage(_Boom(_client_error("404", status=404, operation="HeadObject")))
    with pytest.raises(S3NotFoundError):
        await storage.head_object("uploads/missing")


@pytest.mark.asyncio
async def test_head_access_denied_is_not_missing() -> None:
    storage = _storage(
        _Boom(_client_error("AccessDenied", status=403, operation="HeadObject"))
    )
    with pytest.raises(S3StorageError) as caught:
        await storage.head_object("uploads/secret")
    assert not isinstance(caught.value, S3NotFoundError)
    assert "secret.example" not in str(caught.value)


@pytest.mark.asyncio
async def test_delete_swallows_s3_error_only() -> None:
    storage = _storage(
        _Boom(_client_error("InternalError", status=500, operation="DeleteObject"))
    )
    await storage.delete_object("uploads/a")

    broken = _storage(_Boom(RuntimeError("bug")))
    with pytest.raises(RuntimeError):
        await broken.delete_object("uploads/a")
