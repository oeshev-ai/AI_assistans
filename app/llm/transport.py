"""HTTP-транспорт провайдеров: клиенты, JSON/SSE, health."""

from __future__ import annotations

import asyncio
import json
import logging
import ssl
import time
from collections.abc import AsyncIterator
from types import TracebackType
from typing import Any, Self

import httpx
import structlog

from app.infra.circuit_breaker import (
    APIUnavailableError,
    CircuitBreakerError,
    get_breaker,
    safe_api_call,
)
from app.infra.metrics import API_CALLS, ERRORS, LATENCY
from app.llm.config import AssistAIConfig
from app.llm.errors import AssistAIError, counts_toward_circuit
from app.llm.openweather import OpenWeatherClient
from app.llm.providers import (
    PROXYAPI_CAPABILITIES,
    ProviderConfig,
    ProviderRegistry,
)

logger = structlog.get_logger("AssistAI.transport")

_CHAT_PATH = "chat/completions"
_IMAGES_PATH = "images/generations"
_IMAGES_EDIT_PATH = "images/edits"
_TRANSCRIBE_PATH = "audio/transcriptions"
_SPEECH_PATH = "audio/speech"
_VIDEOS_PATH = "videos"
# Картинка и длинный потоковый ответ могут занимать до двух минут.
_IMAGE_READ_TIMEOUT = 180.0
_STREAM_READ_TIMEOUT = 180.0
_TRANSCRIBE_READ_TIMEOUT = 300.0
_SPEECH_READ_TIMEOUT = 300.0
_VIDEO_READ_TIMEOUT = 300.0
_VIDEO_POLL_INTERVAL = 5.0
_VIDEO_POLL_MAX_SECONDS = 600.0
# 429: не более двух повторов; без Retry-After пауза 1, затем 2 с.
_RATE_LIMIT_RETRIES = 2
_MAX_RETRY_AFTER_SECONDS = 60.0


# URL с access_key не должен попадать в лог при уровне INFO.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


async def _sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


def _provider_label(provider: ProviderConfig) -> str:
    """Имя провайдера в сообщениях об ошибках (для ProxyAPI — прежний текст)."""
    return "ProxyAPI" if provider.id == "proxyapi" else provider.id


def _chat_base(provider: ProviderConfig) -> str:
    """Корень для ``/chat/completions``: ``chat_base_url`` или ``base_url``."""
    return (provider.chat_base_url or provider.base_url).rstrip("/")


def _request_base(provider: ProviderConfig, path: str) -> str:
    """База HTTP: chat/audio → ``chat_base_url``, остальное → ``base_url``."""
    cleaned = path.lstrip("/")
    if (
        cleaned == _CHAT_PATH
        or cleaned.startswith("chat/")
        or cleaned == _SPEECH_PATH
        or cleaned == _TRANSCRIBE_PATH
        or cleaned.startswith("audio/")
    ):
        return _chat_base(provider)
    return provider.base_url.rstrip("/")


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """Секунды из ``Retry-After`` (HTTP-date не разбираем)."""
    raw = response.headers.get("retry-after")
    if raw is None:
        return None
    try:
        value = float(raw.strip())
    except ValueError:
        return None
    return value if value >= 0 else None


def _ssl_context() -> ssl.SSLContext:
    """Контекст проверки TLS по хранилищу ОС, а не только по certifi.

    Kaspersky подменяет сертификат своим корнем из хранилища Windows.
    httpx с certifi такой цепочке не доверяет и рвёт запрос.
    """
    return ssl.create_default_context()


_SSE_SKIP = object()
_SSE_DONE = object()


def _parse_sse_data(line: str) -> Any:
    """Разбирает одну строку SSE.

    ``_SSE_SKIP`` — строку нужно пропустить, ``_SSE_DONE`` — поток закончился,
    словарь — очередной JSON-фрагмент.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith(":"):
        return _SSE_SKIP
    if not stripped.startswith("data:"):
        return _SSE_SKIP
    data = stripped[5:].strip()
    if data == "[DONE]":
        return _SSE_DONE
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        logger.error("Пропущен не-JSON фрагмент потока")
        return _SSE_SKIP
    return payload


def _response_error_text(response: httpx.Response) -> str:
    """Достаёт человекочитаемое сообщение об ошибке из тела ответа ProxyAPI."""
    try:
        body = response.json()
    except ValueError:
        return response.text[:500]

    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if message:
                return str(message)[:500]
        if isinstance(error, str) and error:
            return error[:500]
    return response.text[:500]


class ProviderTransportMixin:
    """Миксин HTTP/registry для ``AssistAIClient``."""

    config: AssistAIConfig
    _registry: ProviderRegistry
    _http: httpx.AsyncClient
    _extra_clients: dict[str, httpx.AsyncClient]
    _closed: bool
    _weather_client: OpenWeatherClient | None

    def _build_registry(self) -> ProviderRegistry:
        """Реестр провайдеров. ProxyAPI всегда из ``base_url``/``proxyapi_key`` (fallback).

        Дефолтный primary — polza, если провайдер зарегистрирован; иначе proxyapi.
        Явные ``config.routes`` перекрывают seed.
        """
        providers = dict(self.config.providers)
        providers["proxyapi"] = ProviderConfig(
            id="proxyapi",
            kind="openai",
            base_url=self.config.base_url,
            api_key=self.config.proxyapi_key,
            capabilities=PROXYAPI_CAPABILITIES,
        )
        default_primary = "polza" if "polza" in providers else "proxyapi"
        routes = {op: default_primary for op in ("text", "image", "audio", "video")}
        routes.update(self.config.routes)
        registry = ProviderRegistry(
            providers,
            routes,
            image_fallback=self.config.image_provider_fallback,
            text_fallback=self.config.text_provider_fallback,
            audio_fallback=self.config.audio_provider_fallback,
            video_fallback=self.config.video_provider_fallback,
        )
        required = {
            "text": "chat",
            "image": "images",
            "audio": "audio",
            "video": "video",
        }
        for operation, capability in required.items():
            provider = registry.get(registry.routes[operation])
            if capability not in provider.capabilities:
                raise ValueError(
                    f"Провайдер '{provider.id}' не поддерживает операцию "
                    f"'{operation}' ({capability})."
                )
        return registry

    def _new_http_client(self, provider: ProviderConfig) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=httpx.Timeout(self.config.timeout),
            follow_redirects=False,
            verify=_ssl_context(),
            headers={
                "Authorization": f"Bearer {provider.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "AssistAIClient/1.0",
            },
        )

    def _endpoint(self, operation: str) -> ProviderConfig:
        """Основной провайдер операции: ``text``/``chat``, ``image``, ``audio``, ``video``."""
        primary, _fallback = self._registry.resolve_route(operation)
        return primary

    def _client_for(self, provider: ProviderConfig) -> httpx.AsyncClient:
        """JSON-клиент провайдера. ProxyAPI — ``self._http``, остальные — лениво."""
        if provider.id == "proxyapi":
            return self._http
        client = self._extra_clients.get(provider.id)
        if client is None:
            client = self._new_http_client(provider)
            self._extra_clients[provider.id] = client
        return client

    def provider_ids(self) -> list[str]:
        """Зарегистрированные провайдеры (для health)."""
        return sorted(self._registry.providers)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        """Закрывает сессию ProxyAPI и ``OpenWeatherClient``, если он был создан.

        Повторный вызов ничего не делает.
        """
        if self._closed:
            return
        self._closed = True
        await self._http.aclose()
        for extra in self._extra_clients.values():
            await extra.aclose()
        self._extra_clients.clear()
        if self._weather_client is not None:
            await self._weather_client.close()
            self._weather_client = None
        logger.info("AssistAIClient закрыт")

    async def _iter_chat_chunks(
        self,
        payload: dict[str, Any],
        *,
        provider: ProviderConfig | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Читает SSE-поток ``/chat/completions`` и отдаёт JSON каждого ``data:``."""
        if self._closed:
            raise AssistAIError("Клиент уже закрыт.")

        target = provider or self._endpoint("text")
        label = _provider_label(target)
        breaker = get_breaker(target.id)
        try:
            await breaker.ensure_allows_call()
        except CircuitBreakerError as exc:
            raise AssistAIError(
                "⚠️ LLM сервис временно недоступен. Попробуйте через минуту.",
                error_code="CIRCUIT_OPEN",
            ) from exc

        url = f"{_chat_base(target)}/{_CHAT_PATH}"
        read_timeout = max(self.config.timeout, _STREAM_READ_TIMEOUT)
        timeout = httpx.Timeout(read_timeout, connect=self.config.timeout)
        logger.info("proxy_api_stream", url=url, path=_CHAT_PATH, provider=target.id)
        t0 = time.perf_counter()
        try:
            async with self._client_for(target).stream(
                "POST", url, json=payload, timeout=timeout
            ) as response:
                await self._raise_for_stream_status(response, _CHAT_PATH, label=label)
                async for line in response.aiter_lines():
                    chunk = _parse_sse_data(line)
                    if chunk is _SSE_SKIP:
                        continue
                    if chunk is _SSE_DONE:
                        break
                    if isinstance(chunk, dict):
                        error = chunk.get("error")
                        if error:
                            if isinstance(error, dict):
                                message = str(error.get("message") or error)
                            else:
                                message = str(error)
                            logger.error("proxy_api_stream_error", detail=message)
                            raise AssistAIError(f"{label}: {message}")
                        yield chunk
            API_CALLS.labels(
                service=target.id, endpoint="chat_stream", status="success"
            ).inc()
            await breaker._on_success()
        except AssistAIError as exc:
            API_CALLS.labels(
                service=target.id, endpoint="chat_stream", status="error"
            ).inc()
            ERRORS.labels(error_type=type(exc).__name__, module=target.id).inc()
            if counts_toward_circuit(exc):
                await breaker._on_failure(exc)
            raise
        except httpx.TimeoutException as exc:
            API_CALLS.labels(
                service=target.id, endpoint="chat_stream", status="error"
            ).inc()
            ERRORS.labels(error_type="TimeoutException", module=target.id).inc()
            await breaker._on_failure(exc)
            logger.error("proxy_api_stream_timeout", path=_CHAT_PATH)
            raise AssistAIError(
                f"Превышено время ожидания ответа {label} ({_CHAT_PATH}).",
                error_code="TIMEOUT",
            ) from exc
        except httpx.RequestError as exc:
            API_CALLS.labels(
                service=target.id, endpoint="chat_stream", status="error"
            ).inc()
            ERRORS.labels(error_type=type(exc).__name__, module=target.id).inc()
            await breaker._on_failure(exc)
            logger.error("proxy_api_stream_network", path=_CHAT_PATH, error=str(exc))
            raise AssistAIError(
                f"Сетевая ошибка при обращении к {label}: {exc}",
                error_code="NETWORK_ERROR",
            ) from exc
        finally:
            LATENCY.labels(service=target.id, endpoint="chat_stream").observe(
                time.perf_counter() - t0
            )

    @staticmethod
    async def _raise_for_stream_status(
        response: httpx.Response,
        path: str,
        *,
        label: str = "ProxyAPI",
    ) -> None:
        if response.status_code < 400:
            return
        raw = await response.aread()
        detail = raw.decode("utf-8", errors="replace")[:500]
        try:
            body = json.loads(raw)
        except ValueError:
            body = None
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict) and error.get("message"):
                detail = str(error["message"])[:500]
            elif isinstance(error, str) and error:
                detail = error[:500]
        logger.error("%s HTTP %s для %s: %s", label, response.status_code, path, detail)
        raise AssistAIError(
            f"{label} вернул HTTP {response.status_code}: {detail}",
            status_code=response.status_code,
        )

    async def probe_provider(self, provider_id: str) -> None:
        """Лёгкая проверка доступности одного провайдера для health check.

        Короткий GET к базе. Ошибка сети или HTTP 5xx считается недоступностью;
        401/404 при живом хосте — ок.
        """
        if self._closed:
            raise AssistAIError("Клиент уже закрыт.")
        provider = self._registry.get(provider_id)
        base = provider.base_url.rstrip("/")
        url = f"{base}/v1/models" if provider.kind == "polza_media" else f"{base}/"
        label = _provider_label(provider)
        try:
            response = await self._client_for(provider).get(
                url, timeout=httpx.Timeout(5.0)
            )
        except httpx.HTTPError as exc:
            raise AssistAIError(f"{label} недоступен: {exc}") from exc
        if response.status_code >= 500:
            raise AssistAIError(f"{label} недоступен: HTTP {response.status_code}")

    async def test_connection(self) -> None:
        """Проверка всех зарегистрированных провайдеров.

        Падает, если недоступен хотя бы один; в тексте ошибки — id провайдеров.
        """
        if self._closed:
            raise AssistAIError("Клиент уже закрыт.")
        failed: list[str] = []
        details: list[str] = []
        for provider_id in self.provider_ids():
            try:
                await self.probe_provider(provider_id)
            except AssistAIError as exc:
                failed.append(provider_id)
                details.append(exc.message)
        if failed:
            raise AssistAIError("; ".join(details))

    async def _post_json(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        timeout: httpx.Timeout | None = None,
        provider: ProviderConfig | None = None,
        operation: str = "text",
    ) -> dict[str, Any]:
        """POST JSON провайдеру операции (по умолчанию — маршрут ``operation``).

        Сетевые и HTTP-ошибки становятся ``AssistAIError``.
        """
        target = provider or self._endpoint(operation)

        def _counts(exc: BaseException) -> bool:
            if isinstance(exc, AssistAIError):
                return counts_toward_circuit(exc)
            return True

        try:
            return await safe_api_call(
                get_breaker(target.id),
                self._post_json_raw,
                path,
                payload,
                timeout=timeout,
                provider=target,
                user_message=(
                    "⚠️ LLM сервис временно недоступен. Попробуйте через минуту."
                ),
                failure_filter=_counts,
            )
        except APIUnavailableError as exc:
            logger.error("llm_unavailable", path=path, error=exc.message)
            raise AssistAIError(exc.message, error_code="CIRCUIT_OPEN") from exc

    async def _post_json_raw(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        timeout: httpx.Timeout | None = None,
        provider: ProviderConfig | None = None,
    ) -> dict[str, Any]:
        """Фактический HTTP POST без circuit breaker (для метрик и breaker)."""
        target = provider or self._endpoint("text")
        service = target.id
        started = time.perf_counter()
        try:
            data = await self._post_json_request(target, path, payload, timeout=timeout)
        except Exception as exc:
            API_CALLS.labels(
                service=service, endpoint="post_json", status="error"
            ).inc()
            ERRORS.labels(error_type=type(exc).__name__, module=service).inc()
            raise
        else:
            API_CALLS.labels(
                service=service, endpoint="post_json", status="success"
            ).inc()
            return data
        finally:
            LATENCY.labels(service=service, endpoint="post_json").observe(
                time.perf_counter() - started
            )

    async def _post_json_request(
        self,
        target: ProviderConfig,
        path: str,
        payload: dict[str, Any],
        *,
        timeout: httpx.Timeout | None = None,
    ) -> dict[str, Any]:
        """POST JSON без метрик: метки ``service`` ставит ``_post_json_raw``."""
        if self._closed:
            raise AssistAIError("Клиент уже закрыт.")

        label = _provider_label(target)
        url = f"{_request_base(target, path)}/{path.lstrip('/')}"
        logger.info("proxy_api_post", url=url, path=path, provider=target.id)
        try:
            response = await self._client_for(target).post(
                url, json=payload, timeout=timeout
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = _response_error_text(exc.response)
            logger.error(
                "proxy_api_http_error",
                status_code=exc.response.status_code,
                path=path,
                detail=detail,
            )
            raise AssistAIError(
                f"{label} вернул HTTP {exc.response.status_code}: {detail}",
                status_code=exc.response.status_code,
                retry_after=_retry_after_seconds(exc.response),
            ) from exc
        except httpx.TimeoutException as exc:
            logger.error("proxy_api_timeout", path=path)
            raise AssistAIError(
                f"Превышено время ожидания ответа {label} ({path}).",
                error_code="TIMEOUT",
            ) from exc
        except httpx.RequestError as exc:
            logger.error("proxy_api_network_error", path=path, error=str(exc))
            raise AssistAIError(
                f"Сетевая ошибка при обращении к {label}: {exc}",
                error_code="NETWORK_ERROR",
            ) from exc

        try:
            data = response.json()
        except ValueError as exc:
            logger.error("proxy_api_non_json", path=path)
            raise AssistAIError(f"{label} вернул ответ не в формате JSON.") from exc

        if not isinstance(data, dict):
            raise AssistAIError(f"{label} вернул неожиданный формат ответа.")

        error = data.get("error")
        if error:
            if isinstance(error, dict):
                message = str(error.get("message") or error)
            else:
                message = str(error)
            logger.error("proxy_api_body_error", path=path, detail=message)
            raise AssistAIError(f"{label}: {message}")
        return data
