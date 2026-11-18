"""LLM spend tracking and daily budget enforcement."""

from __future__ import annotations

import asyncio
import math
import os
import time
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

import structlog
from prometheus_client import Counter, Gauge

logger = structlog.get_logger("cost_tracker")

_billing_user_id: ContextVar[int | None] = ContextVar("billing_user_id", default=None)
_reserved_usd: ContextVar[float] = ContextVar("cost_reserved_usd", default=0.0)

# Atomically reserve ``estimated`` against optional global/user daily caps.
# KEYS[1]=day, KEYS[2]=user (ignored when ARGV[5]=0).
_RESERVE_LUA = """
local estimated = tonumber(ARGV[3])
local ttl = tonumber(ARGV[4])
local global_budget = tonumber(ARGV[1])
local user_budget = tonumber(ARGV[2])
local has_user = tonumber(ARGV[5])
if global_budget > 0 then
  local spent = tonumber(redis.call('GET', KEYS[1]) or '0')
  if spent + estimated > global_budget then
    return 0
  end
end
if user_budget > 0 and has_user == 1 then
  local uspent = tonumber(redis.call('GET', KEYS[2]) or '0')
  if uspent + estimated > user_budget then
    return 0
  end
end
if global_budget > 0 then
  redis.call('INCRBYFLOAT', KEYS[1], estimated)
  redis.call('EXPIRE', KEYS[1], ttl)
end
if user_budget > 0 and has_user == 1 then
  redis.call('INCRBYFLOAT', KEYS[2], estimated)
  redis.call('EXPIRE', KEYS[2], ttl)
end
return 1
"""


def set_billing_user(user_id: int | None) -> None:
    """Bind the current Telegram user for budget attribution."""
    _billing_user_id.set(user_id)


def get_billing_user() -> int | None:
    return _billing_user_id.get()


LLM_SPEND_USD = Counter(
    "llm_spend_usd_total",
    "Estimated LLM spend in USD",
    ["kind"],
)

LLM_BUDGET_REMAINING = Gauge(
    "llm_daily_budget_remaining_usd",
    "Remaining global daily LLM budget in USD",
)

# Approximate ProxyAPI / OpenAI-class rates (USD).
CHAT_INPUT_PER_1K = float(os.getenv("COST_CHAT_INPUT_PER_1K", "0.00015"))
CHAT_OUTPUT_PER_1K = float(os.getenv("COST_CHAT_OUTPUT_PER_1K", "0.0006"))
IMAGE_USD = float(os.getenv("COST_IMAGE_USD", "0.04"))
IMAGE_FLUX_USD = float(os.getenv("COST_IMAGE_FLUX_USD", str(IMAGE_USD)))
IMAGE_OUT_USD_PER_1M = float(os.getenv("IMAGE_OUT_USD_PER_1M", "30"))
IMAGE_IN_USD_PER_1M = float(os.getenv("IMAGE_IN_USD_PER_1M", "8"))
TEXT_IN_USD_PER_1M = float(os.getenv("TEXT_IN_USD_PER_1M", "5"))
TEXT_OUT_USD_PER_1M = float(os.getenv("TEXT_OUT_USD_PER_1M", "10"))
# Выходные токены GPT Image. Измерено: 1024²/low=824, 1024×1536 и 1536×1024/high=1353.
# Остальные пары — оценка калькулятора, не проверяли. Ориентация токены не меняет.
_IMAGE_OUTPUT_TOKENS: dict[tuple[str, str], int] = {
    ("1024x1024", "low"): 824,
    ("1024x1024", "medium"): 1449,  # оценка
    ("1024x1024", "high"): 1756,  # оценка калькулятора, не проверяли
    ("1024x1024", "xhigh"): 3122,  # оценка
    ("1024x1024", "max"): 7024,  # оценка
    ("1024x1536", "low"): 664,  # оценка
    ("1024x1536", "medium"): 1132,  # оценка
    ("1024x1536", "high"): 1353,
    ("1024x1536", "xhigh"): 2459,  # оценка
    ("1024x1536", "max"): 5488,  # оценка
}
IMAGE_INPUT_TOKEN_CAP = 1536  # наблюдение: 12 MP → 1452 (44×33); берём верхнюю границу
VIDEO_USD = float(os.getenv("COST_VIDEO_USD", "0.20"))
STT_PER_MINUTE = float(os.getenv("COST_STT_PER_MINUTE", "0.006"))
TTS_PER_1K_CHARS = float(os.getenv("COST_TTS_PER_1K_CHARS", "0.015"))


def _openai_image_model(model: str) -> bool:
    return model.strip().casefold().startswith("openai/")


def estimate_input_image_tokens(width: int, height: int) -> int:
    """Входные image-токены: ceil(w/32)×ceil(h/32), не выше наблюдаемого потолка."""
    patches = math.ceil(max(width, 1) / 32) * math.ceil(max(height, 1) / 32)
    return min(patches, IMAGE_INPUT_TOKEN_CAP)


def _output_size_class(size: str | None) -> str:
    """Ближайший класс таблицы: квадрат или прямоугольник 1024×1536."""
    if not isinstance(size, str) or "x" not in size.casefold():
        return "1024x1024"
    left, _, right = size.lower().partition("x")
    try:
        width = int(left)
        height = int(right)
    except ValueError:
        return "1024x1024"
    if width <= 0 or height <= 0:
        return "1024x1024"
    if abs(width - height) / max(width, height) <= 0.05:
        return "1024x1024"
    return "1024x1536"


def _image_output_tokens(quality: str | None, size: str | None) -> int | None:
    if not quality:
        return None
    return _IMAGE_OUTPUT_TOKENS.get(
        (_output_size_class(size), quality.strip().casefold())
    )


def _token_count(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def cost_from_usage(usage: dict[str, Any]) -> float:
    """Стоимость по модальностям. ``output_tokens`` целиком не умножается на 30."""
    raw_in = usage.get("input_tokens_details") or {}
    raw_out = usage.get("output_tokens_details") or {}
    details_in = raw_in if isinstance(raw_in, dict) else {}
    details_out = raw_out if isinstance(raw_out, dict) else {}
    text_in = _token_count(details_in.get("text_tokens", usage.get("input_tokens", 0)))
    image_in = _token_count(details_in.get("image_tokens", 0))
    image_out = _token_count(
        details_out.get(
            "image_tokens",
            usage.get("output_tokens", usage.get("completion_tokens", 0)),
        )
    )
    text_out = _token_count(details_out.get("text_tokens", 0))
    if text_out:
        logger.warning("image_text_output_tokens", tokens=text_out)
    return (
        text_in * TEXT_IN_USD_PER_1M
        + image_in * IMAGE_IN_USD_PER_1M
        + image_out * IMAGE_OUT_USD_PER_1M
        + text_out * TEXT_OUT_USD_PER_1M
    ) / 1_000_000


def _image_cost_from_usage(usage: dict[str, Any] | None) -> float | None:
    if not isinstance(usage, dict):
        return None
    recognized = {
        "input_tokens",
        "output_tokens",
        "completion_tokens",
        "input_tokens_details",
        "output_tokens_details",
    }
    if not recognized.intersection(usage):
        return None
    return cost_from_usage(usage)


class BudgetExceededError(Exception):
    """Raised when a call would exceed the configured daily budget."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class CostTracker:
    """Tracks estimated USD spend with Redis durability and memory fallback."""

    def __init__(self) -> None:
        self._redis: Any | None = None
        self._memory: dict[str, float] = {}
        self._lock = asyncio.Lock()
        self.daily_budget = float(os.getenv("LLM_DAILY_BUDGET_USD", "0") or 0)
        self.user_daily_budget = float(os.getenv("LLM_USER_DAILY_BUDGET_USD", "0") or 0)

    def attach_redis(self, redis_client: Any | None) -> None:
        self._redis = redis_client
        if redis_client is not None:
            logger.info("cost_tracker_using_redis")
        else:
            logger.warning("cost_tracker_using_memory")

    @staticmethod
    def _day_key() -> str:
        return datetime.now(UTC).strftime("%Y%m%d")

    def estimate_chat_cost(
        self,
        *,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        input_chars: int = 0,
        output_chars: int = 0,
    ) -> float:
        """Estimate chat USD from provider usage or a chars/4 heuristic."""
        if prompt_tokens is None:
            prompt_tokens = max(1, input_chars // 4) if input_chars else 0
        if completion_tokens is None:
            completion_tokens = max(1, output_chars // 4) if output_chars else 0
        return (
            prompt_tokens / 1000.0 * CHAT_INPUT_PER_1K
            + completion_tokens / 1000.0 * CHAT_OUTPUT_PER_1K
        )

    @staticmethod
    def estimate_image_cost(
        count: int = 1,
        *,
        model: str | None = None,
        quality: str | None = None,
        size: str | None = None,
        usage: dict[str, Any] | None = None,
    ) -> float:
        """Стоимость картинки: usage, затем таблица quality, иначе плоская ставка."""
        copies = max(1, count)
        if model and not _openai_image_model(model):
            return IMAGE_FLUX_USD * copies
        from_usage = _image_cost_from_usage(usage)
        if from_usage is not None:
            return from_usage
        tokens = _image_output_tokens(quality, size)
        if tokens is None:
            return IMAGE_USD * copies
        return copies * tokens * IMAGE_OUT_USD_PER_1M / 1_000_000

    @staticmethod
    def estimate_stt_cost(duration_seconds: float) -> float:
        return max(0.0, duration_seconds) / 60.0 * STT_PER_MINUTE

    @staticmethod
    def estimate_tts_cost(text: str) -> float:
        return max(1, len(text)) / 1000.0 * TTS_PER_1K_CHARS

    async def check_budget(
        self, user_id: int | None = None, estimated: float = 0.0
    ) -> None:
        """Atomically reserve ``estimated`` against daily caps or raise.

        Reservation is settled in ``record`` (delta = actual − reserved).
        """
        if estimated <= 0:
            return
        if self.daily_budget <= 0 and self.user_daily_budget <= 0:
            return
        if user_id is None:
            user_id = get_billing_user()
        day = self._day_key()
        day_key = f"cost:day:{day}"
        user_key = (
            f"cost:user:{user_id}:day:{day}"
            if user_id is not None and self.user_daily_budget > 0
            else ""
        )
        ok = await self._reserve(
            day_key=day_key,
            user_key=user_key,
            estimated=estimated,
            ttl_seconds=3 * 86400,
        )
        if not ok:
            if self.daily_budget > 0:
                spent = await self._get(day_key)
                if spent + estimated > self.daily_budget:
                    logger.warning(
                        "global_budget_exceeded",
                        spent=spent,
                        budget=self.daily_budget,
                        estimated=estimated,
                    )
                    raise BudgetExceededError(
                        "⚠️ Дневной лимит расходов на LLM исчерпан. Попробуйте завтра."
                    )
            logger.warning(
                "user_budget_exceeded",
                user_id=user_id,
                budget=self.user_daily_budget,
                estimated=estimated,
            )
            raise BudgetExceededError(
                "⚠️ Ваш дневной лимит запросов к LLM исчерпан. Попробуйте завтра."
            )
        _reserved_usd.set(_reserved_usd.get() + estimated)
        if self.daily_budget > 0:
            spent = await self._get(day_key)
            LLM_BUDGET_REMAINING.set(max(0.0, self.daily_budget - spent))

    async def record(
        self,
        *,
        kind: str,
        amount_usd: float,
        user_id: int | None = None,
    ) -> None:
        """Persist spend and update Prometheus counters.

        Clears any ``check_budget`` reservation first, then charges the actual
        amount so counters stay accurate under concurrent turns.
        """
        if amount_usd < 0:
            return
        if user_id is None:
            user_id = get_billing_user()
        day = self._day_key()
        ttl = 3 * 86400
        reserved = _reserved_usd.get()
        if reserved > 0:
            _reserved_usd.set(0.0)
            if self.daily_budget > 0:
                await self._incr(f"cost:day:{day}", -reserved, ttl_seconds=ttl)
            if user_id is not None and self.user_daily_budget > 0:
                await self._incr(
                    f"cost:user:{user_id}:day:{day}",
                    -reserved,
                    ttl_seconds=ttl,
                )
        if amount_usd <= 0:
            return
        await self._incr(f"cost:day:{day}", amount_usd, ttl_seconds=ttl)
        if user_id is not None:
            await self._incr(
                f"cost:user:{user_id}:day:{day}",
                amount_usd,
                ttl_seconds=ttl,
            )
        LLM_SPEND_USD.labels(kind=kind).inc(amount_usd)
        if self.daily_budget > 0:
            spent = await self._get(f"cost:day:{day}")
            LLM_BUDGET_REMAINING.set(max(0.0, self.daily_budget - spent))
        logger.info(
            "llm_spend_recorded",
            kind=kind,
            amount_usd=round(amount_usd, 6),
            user_id=user_id,
            reserved_cleared=round(reserved, 6),
        )

    async def _reserve(
        self,
        *,
        day_key: str,
        user_key: str,
        estimated: float,
        ttl_seconds: int,
    ) -> bool:
        if self._redis is not None:
            try:
                has_user = 1 if user_key else 0
                result = await self._redis.eval(
                    _RESERVE_LUA,
                    2,
                    day_key,
                    user_key or day_key,
                    self.daily_budget,
                    self.user_daily_budget if user_key else 0.0,
                    estimated,
                    ttl_seconds,
                    has_user,
                )
                return int(result) == 1
            except Exception as exc:
                logger.warning(
                    "cost_tracker_redis_reserve_failed",
                    error=str(exc),
                    fallback="memory",
                )
        async with self._lock:
            if self.daily_budget > 0:
                spent = float(self._memory.get(day_key, 0.0))
                if spent + estimated > self.daily_budget:
                    return False
            if user_key and self.user_daily_budget > 0:
                uspent = float(self._memory.get(user_key, 0.0))
                if uspent + estimated > self.user_daily_budget:
                    return False
            if self.daily_budget > 0:
                self._memory[day_key] = (
                    float(self._memory.get(day_key, 0.0)) + estimated
                )
            if user_key and self.user_daily_budget > 0:
                self._memory[user_key] = (
                    float(self._memory.get(user_key, 0.0)) + estimated
                )
            return True

    async def _get(self, key: str) -> float:
        if self._redis is not None:
            try:
                raw = await self._redis.get(key)
                return float(raw) if raw is not None else 0.0
            except Exception as exc:
                logger.warning("cost_tracker_redis_get_failed", error=str(exc))
        return float(self._memory.get(key, 0.0))

    async def _incr(self, key: str, amount: float, *, ttl_seconds: int) -> None:
        if self._redis is not None:
            try:
                pipe = self._redis.pipeline()
                pipe.incrbyfloat(key, amount)
                pipe.expire(key, ttl_seconds)
                await pipe.execute()
                return
            except Exception as exc:
                logger.warning("cost_tracker_redis_incr_failed", error=str(exc))
        async with self._lock:
            self._memory[key] = self._memory.get(key, 0.0) + amount
        # Soft expiry for memory mode: drop keys older than ~3 days by day stamp.
        cutoff = time.time() - ttl_seconds
        _ = cutoff


cost_tracker = CostTracker()
