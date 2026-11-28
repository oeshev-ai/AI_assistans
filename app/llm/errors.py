"""Ошибки LLM-слоя и классификация сбоев для fallback между провайдерами.

``AssistAIError`` вынесен сюда, чтобы ``polza_media`` мог его поднимать без
циклического импорта с ``assist_ai`` (тот по-прежнему реэкспортирует класс).
"""

from __future__ import annotations

# Коды, при которых провайдер считается недоступным: допустим один fallback.
FALLBACK_CODES: dict[str, str] = {
    "TIMEOUT": "timeout",
    "REQUEST_TIMEOUT": "request_timeout",
    "SERVICE_UNAVAILABLE": "service_unavailable",
    "BAD_GATEWAY": "bad_gateway",
    "INTERNAL_ERROR": "internal_error",
    "NETWORK_ERROR": "network_error",
    "CIRCUIT_OPEN": "circuit_open",
    "POLL_TIMEOUT": "poll_timeout",
    "TOO_MANY_REQUESTS": "rate_limited",
    "INSUFFICIENT_BALANCE": "insufficient_balance",
}

# Клиентские ошибки: ни повтора, ни fallback (смена провайдера не поможет).
NO_FALLBACK_CODES = frozenset(
    {
        "BAD_REQUEST",
        "UNAUTHORIZED",
        "FORBIDDEN",
        "NOT_FOUND",
        "PAYLOAD_TOO_LARGE",
        "API_KEY_REVOKED",
        "UNOFFICIAL_PROVIDERS_DISABLED",
        "INPUT_TOO_LARGE",
    }
)
NO_FALLBACK_STATUSES = frozenset({400, 401, 403, 404, 413})

# Платный one-shot (картинка/видео/аудио): таймаут/сеть после ухода запроса
# могли уже создать задачу у провайдера — второй хост = риск двойной оплаты.
PAID_AMBIGUOUS_CODES = frozenset(
    {
        "TIMEOUT",
        "REQUEST_TIMEOUT",
        "NETWORK_ERROR",
    }
)


class AssistAIError(Exception):
    """Ошибка запроса к провайдеру или исчерпанный лимит вызовов инструментов.

    Сбой самого инструмента сюда не попадает: он возвращается модели
    как JSON ``{"error": "..."}``.

    ``error_code`` — нормализованный код (``SERVICE_UNAVAILABLE``, ``TIMEOUT``,
    ``CIRCUIT_OPEN`` …), ``retry_after`` — пауза из заголовка ``Retry-After``.
    ``job_committed`` — задача уже принята провайдером (poll/download); fallback
    на другого провайдера запрещён, чтобы не платить дважды.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_code: str | None = None,
        retry_after: float | None = None,
        job_committed: bool = False,
        pending_url: str | None = None,
        pending_provider: str | None = None,
        pending_job_id: str | None = None,
        pending_kind: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.retry_after = retry_after
        self.job_committed = job_committed
        # Уже оплаченный файл. Ссылка и id задачи в текст ошибки и в лог не попадают.
        self.pending_url = pending_url
        self.pending_provider = pending_provider
        self.pending_job_id = pending_job_id
        self.pending_kind = pending_kind


def _normalized_code(exc: AssistAIError) -> str:
    return (exc.error_code or "").strip().upper()


def is_rate_limited(exc: AssistAIError) -> bool:
    """429 / ``TOO_MANY_REQUESTS``: допускается пауза и повтор."""
    return _normalized_code(exc) == "TOO_MANY_REQUESTS" or exc.status_code == 429


def fallback_reason(
    exc: AssistAIError, *, paid_side_effect: bool = False
) -> str | None:
    """Причина для fallback или ``None``, если переключаться нельзя.

    Неизвестные ошибки трактуются консервативно: без fallback.
    ``paid_side_effect=True`` — для картинок/видео/аудио: TIMEOUT/NETWORK
    без ``job_committed`` тоже без fallback (запрос мог уже приняться).
    """
    if exc.job_committed:
        return None
    code = _normalized_code(exc)
    if code.startswith("GUARD_") or code in NO_FALLBACK_CODES:
        return None
    if paid_side_effect and code in PAID_AMBIGUOUS_CODES:
        return None
    if code in FALLBACK_CODES:
        return FALLBACK_CODES[code]
    status = exc.status_code
    if status is None:
        return None
    if status in NO_FALLBACK_STATUSES:
        return None
    if status == 408:
        if paid_side_effect:
            return None
        return "request_timeout"
    if status == 429:
        return "rate_limited"
    if status == 402:
        return "insufficient_balance"
    if status >= 500:
        return "http_5xx"
    return None


def counts_toward_circuit(exc: AssistAIError) -> bool:
    """True if the error should open/trip a provider circuit breaker.

    Client 4xx and rate limits do not indicate provider outage; only
    availability-class failures (5xx, timeouts, network, balance, etc.).
    """
    reason = fallback_reason(exc)
    return reason is not None and reason != "rate_limited"
