"""Реестр LLM-провайдеров и маршрутизация операций."""

from __future__ import annotations

import os
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator

_LOCAL_HTTP_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def require_secure_base_url(value: str) -> str:
    """Normalize provider base URL; ban cleartext remote URLs in production."""
    normalized = value.strip().rstrip("/")
    if not normalized.startswith(("http://", "https://")):
        raise ValueError("base_url должен начинаться с http:// или https://")
    if normalized.startswith("http://"):
        host = (urlparse(normalized).hostname or "").casefold()
        if host not in _LOCAL_HTTP_HOSTS:
            from app.infra.access import is_production

            if is_production():
                raise ValueError(
                    "base_url должен использовать https:// в production "
                    "(http:// разрешён только для localhost)"
                )
    return normalized


class ProviderConfig(BaseModel):
    """Описание одного провайдера. Ключ не попадает в ``repr``."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str
    kind: Literal["openai", "polza_media"]
    base_url: str
    api_key: str = Field(repr=False)
    capabilities: frozenset[str]
    cost_source: Literal["usage_tokens", "cost_rub"] = "usage_tokens"
    cost_multiplier: float = 1.0
    upstream_only: tuple[str, ...] = ()
    allow_upstream_fallbacks: bool = True
    max_price_image_rub: float = 0.0
    # OpenAI-совместимый корень для chat (Polza: …/api/v1). Пусто → base_url.
    chat_base_url: str = ""

    @field_validator("base_url")
    @classmethod
    def _normalize_base_url(cls, value: str) -> str:
        return require_secure_base_url(value)

    @field_validator("chat_base_url")
    @classmethod
    def _normalize_chat_base_url(cls, value: str) -> str:
        cleaned = (value or "").strip().rstrip("/")
        if not cleaned:
            return ""
        return require_secure_base_url(cleaned)

    @field_validator("capabilities", mode="before")
    @classmethod
    def _coerce_capabilities(cls, value: Any) -> Any:
        if isinstance(value, (list, tuple, set)):
            return frozenset(str(item) for item in value)
        return value

    def media_provider_object(
        self,
        *,
        pin_upstream: bool = True,
        send_max_price: bool = True,
    ) -> dict[str, Any] | None:
        """Объект ``provider`` для Polza ``/v1/media`` (пиннинг апстрима, потолок цены).

        ``require_parameters`` и ``data_residency`` намеренно не отправляются:
        первый превратил бы ``warnings[]`` в ошибки, второй отклонит OpenAI-модели.

        ``pin_upstream=False`` — без ``only`` (FLUX/Qwen: ``only=["OpenAI"]`` →
        ``noProvidersForMedia``). ``send_max_price=False`` — без потолка: у
        token-priced ``gpt-image`` и у klein с ``unitParam`` Polza отвечает 400
        «цена не определена».
        """
        obj: dict[str, Any] = {}
        if pin_upstream and self.upstream_only:
            obj["only"] = list(self.upstream_only)
            obj["allow_fallbacks"] = bool(self.allow_upstream_fallbacks)
        elif pin_upstream and not self.allow_upstream_fallbacks:
            obj["allow_fallbacks"] = False
        if send_max_price and self.max_price_image_rub > 0:
            obj["max_price"] = {"image": self.max_price_image_rub}
        return obj or None


_ROUTE_OPS = ("text", "image", "audio", "video")
_DEFAULT_PROXYAPI_BASE = "https://api.proxyapi.ru/v1"
_DEFAULT_POLZA_BASE = "https://polza.ai/api"
_DEFAULT_POLZA_CHAT_BASE = "https://polza.ai/api/v1"
PROXYAPI_CAPABILITIES = frozenset({"chat", "images", "images_edit", "audio", "video"})
_PROXYAPI_CAPABILITIES = PROXYAPI_CAPABILITIES
_POLZA_CAPABILITIES = frozenset({"chat", "images", "images_edit"})


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().casefold() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _split_csv(raw: str) -> tuple[str, ...]:
    if not raw:
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def load_providers(
    *,
    proxyapi_key: str | None = None,
    polza_key: str | None = None,
) -> dict[str, ProviderConfig]:
    """Собирает реестр из env. Без ключа провайдер не регистрируется."""
    providers: dict[str, ProviderConfig] = {}
    proxy_key = proxyapi_key if proxyapi_key is not None else _env("PROXYAPI_API_KEY")
    if proxy_key:
        providers["proxyapi"] = ProviderConfig(
            id="proxyapi",
            kind="openai",
            base_url=_env("PROXYAPI_BASE_URL", _DEFAULT_PROXYAPI_BASE)
            or _DEFAULT_PROXYAPI_BASE,
            api_key=proxy_key,
            capabilities=_PROXYAPI_CAPABILITIES,
            cost_source="usage_tokens",
            cost_multiplier=1.0,
        )

    polza = polza_key if polza_key is not None else _env("POLZA_API_KEY")
    if polza:
        upstream = _split_csv(_env("POLZA_UPSTREAM_ONLY", "OpenAI"))
        providers["polza"] = ProviderConfig(
            id="polza",
            kind="polza_media",
            base_url=_env("POLZA_BASE_URL", _DEFAULT_POLZA_BASE) or _DEFAULT_POLZA_BASE,
            api_key=polza,
            capabilities=_POLZA_CAPABILITIES,
            cost_source="cost_rub",
            cost_multiplier=_env_float("POLZA_COST_MULTIPLIER", 0.39),
            upstream_only=upstream or ("OpenAI",),
            allow_upstream_fallbacks=_env_flag("POLZA_ALLOW_UPSTREAM_FALLBACKS", False),
            max_price_image_rub=_env_float("POLZA_MAX_PRICE_IMAGE_RUB", 12.0),
            chat_base_url=_env("POLZA_CHAT_BASE_URL", _DEFAULT_POLZA_CHAT_BASE)
            or _DEFAULT_POLZA_CHAT_BASE,
        )
    return providers


def load_routes() -> dict[str, str]:
    """Маршруты операций → id провайдера. Пустые env → proxyapi."""
    return {
        "text": _env("TEXT_PROVIDER", "proxyapi") or "proxyapi",
        "image": _env("IMAGE_PROVIDER", "proxyapi") or "proxyapi",
        "audio": _env("AUDIO_PROVIDER", "proxyapi") or "proxyapi",
        "video": _env("VIDEO_PROVIDER", "proxyapi") or "proxyapi",
    }


def load_image_fallback() -> str:
    return _env("IMAGE_PROVIDER_FALLBACK", "proxyapi")


class ProviderRegistry:
    """Реестр + resolve_route с проверкой неизвестных id на старте."""

    def __init__(
        self,
        providers: dict[str, ProviderConfig],
        routes: dict[str, str] | None = None,
        *,
        image_fallback: str = "",
    ) -> None:
        self.providers = dict(providers)
        self.routes = dict(routes or {op: "proxyapi" for op in _ROUTE_OPS})
        self.image_fallback = (image_fallback or "").strip()
        self._validate()

    @classmethod
    def from_env(
        cls,
        *,
        proxyapi_key: str | None = None,
        polza_key: str | None = None,
    ) -> ProviderRegistry:
        return cls(
            load_providers(proxyapi_key=proxyapi_key, polza_key=polza_key),
            load_routes(),
            image_fallback=load_image_fallback(),
        )

    def _validate(self) -> None:
        for op, provider_id in self.routes.items():
            if provider_id not in self.providers:
                known = ", ".join(sorted(self.providers)) or "(пусто)"
                raise ValueError(
                    f"Неизвестный провайдер '{provider_id}' в маршруте "
                    f"'{op}'. Зарегистрированы: {known}"
                )
        if self.image_fallback and self.image_fallback not in self.providers:
            # Пустой fallback допустим; непустой неизвестный — ошибка старта.
            # Но если IMAGE_PROVIDER_FALLBACK=proxyapi при отсутствии ключа
            # proxyapi — тоже ошибка. Приёмка G1: без новых env только proxyapi,
            # fallback по умолчанию proxyapi совпадает с основным → ок.
            known = ", ".join(sorted(self.providers)) or "(пусто)"
            raise ValueError(
                f"Неизвестный IMAGE_PROVIDER_FALLBACK "
                f"'{self.image_fallback}'. Зарегистрированы: {known}"
            )

    def get(self, provider_id: str) -> ProviderConfig:
        try:
            return self.providers[provider_id]
        except KeyError as exc:
            raise KeyError(f"Провайдер '{provider_id}' не зарегистрирован") from exc

    def resolve_route(
        self, operation: str
    ) -> tuple[ProviderConfig, ProviderConfig | None]:
        """Основной провайдер и опциональный fallback (только для image)."""
        op = operation.strip().casefold()
        # aliases for call sites
        if op in {"images", "images_edit", "image_edit", "edit"}:
            op = "image"
        if op in {"chat", "text"}:
            op = "text"
        if op not in self.routes:
            raise ValueError(f"Неизвестная операция маршрута: {operation}")
        primary_id = self.routes[op]
        primary = self.get(primary_id)
        fallback: ProviderConfig | None = None
        if op == "image" and self.image_fallback and self.image_fallback != primary_id:
            fallback = self.get(self.image_fallback)
        return primary, fallback
