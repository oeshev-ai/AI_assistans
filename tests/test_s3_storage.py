"""S3-слой ловит ошибки хранилища и не маскирует сбои кода."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.storage.s3_storage import (
    TRANSCRIPT_PREFIX,
    TRANSCRIPT_RETENTION_DAYS,
    TRANSCRIPT_URL_SECONDS,
    S3Config,
    S3NotFoundError,
    S3Storage,
    S3StorageError,
)


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


class _Body:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._offset = 0
        self.closed = False

    def read(self, size: int) -> bytes:
        if self._offset >= len(self._payload):
            return b""
        part = self._payload[self._offset : self._offset + size]
        self._offset += len(part)
        return part

    def close(self) -> None:
        self.closed = True


class _Get:
    def __init__(self, payload: bytes, *, content_length: int | None = None) -> None:
        self.body = _Body(payload)
        self._length = len(payload) if content_length is None else content_length

    def get_object(self, **_kwargs: Any) -> dict[str, Any]:
        return {"ContentLength": self._length, "Body": self.body}


@pytest.mark.asyncio
async def test_download_file_streams_without_joining(tmp_path: Path) -> None:
    payload = b"abcdefghij"
    client = _Get(payload)
    storage = _storage(client)
    destination = tmp_path / "object.bin"
    await storage.download_file("uploads/a", destination, max_bytes=100)
    assert destination.read_bytes() == payload
    assert client.body.closed


class _Signer:
    def __init__(self) -> None:
        self.operation = ""
        self.kwargs: dict[str, Any] = {}

    def generate_presigned_url(self, operation: str, **kwargs: Any) -> str:
        self.operation = operation
        self.kwargs = kwargs
        return (
            "https://storage.yandexcloud.net/bucket/transcripts/a.txt"
            "?X-Amz-Signature=abc"
        )


@pytest.mark.asyncio
async def test_presign_get_signs_download_without_leaking_the_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.storage.s3_endpoint.guard_s3_request", lambda *_a, **_k: None
    )
    client = _Signer()
    storage = _storage(client)
    url = await storage.create_presigned_get(
        "transcripts/a.txt", expires_in=TRANSCRIPT_URL_SECONDS
    )
    assert client.operation == "get_object"
    assert client.kwargs["ExpiresIn"] == TRANSCRIPT_URL_SECONDS
    assert client.kwargs["HttpMethod"] == "GET"
    assert client.kwargs["Params"]["Key"] == "transcripts/a.txt"
    assert "X-Amz-Signature=abc" in url


@pytest.mark.asyncio
async def test_presign_get_wraps_s3_error_without_its_text() -> None:
    storage = _storage(
        _Boom(_client_error("AccessDenied", status=403, operation="GetObject"))
    )
    with pytest.raises(S3StorageError) as caught:
        await storage.create_presigned_get("transcripts/a.txt", expires_in=60)
    assert "secret.example" not in str(caught.value)


class _Lifecycle:
    def __init__(
        self,
        rules: list[dict[str, Any]] | None = None,
        get_error: BaseException | None = None,
    ) -> None:
        self._rules = list(rules or [])
        self._get_error = get_error
        self.gets = 0
        self.put: dict[str, Any] | None = None

    def get_bucket_lifecycle_configuration(self, **_kwargs: Any) -> dict[str, Any]:
        self.gets += 1
        if self._get_error is not None:
            raise self._get_error
        return {"Rules": self._rules}

    def put_bucket_lifecycle_configuration(self, **kwargs: Any) -> None:
        self.put = kwargs


def _put_rules(client: _Lifecycle) -> list[dict[str, Any]]:
    assert client.put is not None
    rules = client.put["LifecycleConfiguration"]["Rules"]
    assert isinstance(rules, list)
    return rules


@pytest.mark.asyncio
async def test_transcript_lifecycle_is_created_when_the_bucket_has_none() -> None:
    client = _Lifecycle(
        get_error=_client_error(
            "NoSuchLifecycleConfiguration", status=404, operation="GetBucketLifecycle"
        )
    )
    storage = _storage(client)
    await storage.ensure_transcript_lifecycle()
    rules = _put_rules(client)
    assert rules == [
        {
            "ID": "expire-transcripts",
            "Status": "Enabled",
            "Filter": {"Prefix": TRANSCRIPT_PREFIX},
            "Expiration": {"Days": TRANSCRIPT_RETENTION_DAYS},
        }
    ]
    await storage.ensure_transcript_lifecycle()
    assert client.gets == 1


@pytest.mark.asyncio
async def test_transcript_lifecycle_keeps_other_rules() -> None:
    foreign = {
        "ID": "expire-uploads",
        "Status": "Enabled",
        "Filter": {"Prefix": "uploads/"},
        "Expiration": {"Days": 1},
        "ResponseMetadata": {"secret": "https://secret.example/internal"},
    }
    client = _Lifecycle(rules=[foreign])
    storage = _storage(client)
    await storage.ensure_transcript_lifecycle()
    rules = _put_rules(client)
    assert rules[0] == {
        "ID": "expire-uploads",
        "Status": "Enabled",
        "Filter": {"Prefix": "uploads/"},
        "Expiration": {"Days": 1},
    }
    assert rules[1]["Filter"] == {"Prefix": "transcripts/"}
    assert rules[1]["Expiration"] == {"Days": 7}
    assert "secret.example" not in str(client.put)


@pytest.mark.asyncio
async def test_transcript_lifecycle_is_not_rewritten_when_it_matches() -> None:
    current = {
        "ID": "expire-transcripts",
        "Status": "Enabled",
        "Filter": {"Prefix": "transcripts/"},
        "Expiration": {"Days": 7},
    }
    client = _Lifecycle(rules=[current])
    storage = _storage(client)
    await storage.ensure_transcript_lifecycle()
    await storage.ensure_transcript_lifecycle()
    assert client.put is None
    assert client.gets == 1


@pytest.mark.asyncio
async def test_transcript_lifecycle_replaces_a_different_transcript_rule() -> None:
    stale = {
        "ID": "expire-transcripts",
        "Status": "Enabled",
        "Filter": {"Prefix": "transcripts/"},
        "Expiration": {"Days": 365},
    }
    client = _Lifecycle(rules=[stale])
    await _storage(client).ensure_transcript_lifecycle()
    rules = _put_rules(client)
    assert len(rules) == 1
    assert rules[0]["Expiration"] == {"Days": TRANSCRIPT_RETENTION_DAYS}


@pytest.mark.asyncio
async def test_transcript_lifecycle_does_not_put_when_the_read_fails() -> None:
    client = _Lifecycle(
        get_error=_client_error(
            "AccessDenied", status=403, operation="GetBucketLifecycle"
        )
    )
    with pytest.raises(S3StorageError) as caught:
        await _storage(client).ensure_transcript_lifecycle()
    assert client.put is None
    assert "secret.example" not in str(caught.value)


@pytest.mark.asyncio
async def test_transcript_lifecycle_lets_programming_errors_through() -> None:
    client = _Lifecycle(get_error=RuntimeError("bug"))
    with pytest.raises(RuntimeError):
        await _storage(client).ensure_transcript_lifecycle()
    assert client.put is None


@pytest.mark.asyncio
async def test_download_file_over_limit_removes_partial(tmp_path: Path) -> None:
    client = _Get(b"abcdefghij", content_length=0)
    storage = _storage(client)
    destination = tmp_path / "object.bin"
    destination.write_bytes(b"partial")
    with pytest.raises(S3StorageError):
        await storage.download_file("uploads/a", destination, max_bytes=4)
    assert not destination.exists()
    assert client.body.closed
