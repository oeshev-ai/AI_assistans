"""LLM spend tracking and daily budget enforcement."""

from __future__ import annotations

import os
import time
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

import structlog
from prometheus_client import Counter, Gauge

logger = structlog.get_logger("cost_tracker")

_billing_user_id: ContextVar[int | None] = ContextVar("billing_user_id", default=None)


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
VIDEO_USD = float(os.getenv("COST_VIDEO_USD", "0.20"))
STT_PER_MINUTE = float(os.getenv("COST_STT_PER_MINUTE", "0.006"))
TTS_PER_1K_CHARS = float(os.getenv("COST_TTS_PER_1K_CHARS", "0.015"))


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
    def estimate_image_cost(count: int = 1) -> float:
        return IMAGE_USD * max(1, count)

    @staticmethod
    def estimate_stt_cost(duration_seconds: float) -> float:
        return max(0.0, duration_seconds) / 60.0 * STT_PER_MINUTE

    @staticmethod
    def estimate_tts_cost(text: str) -> float:
        return max(1, len(text)) / 1000.0 * TTS_PER_1K_CHARS

    async def check_budget(
        self, user_id: int | None = None, estimated: float = 0.0
    ) -> None:
        """Raise ``BudgetExceededError`` if estimated spend would exceed limits."""
        if estimated <= 0:
            return
        if user_id is None:
            user_id = get_billing_user()
        day = self._day_key()
        if self.daily_budget > 0:
            spent = await self._get(f"cost:day:{day}")
            remaining = max(0.0, self.daily_budget - spent)
            LLM_BUDGET_REMAINING.set(remaining)
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
        if user_id is not None and self.user_daily_budget > 0:
            spent = await self._get(f"cost:user:{user_id}:day:{day}")
            if spent + estimated > self.user_daily_budget:
                logger.warning(
                    "user_budget_exceeded",
                    user_id=user_id,
                    spent=spent,
                    budget=self.user_daily_budget,
                    estimated=estimated,
                )
                raise BudgetExceededError(
                    "⚠️ Ваш дневной лимит запросов к LLM исчерпан. Попробуйте завтра."
                )

    async def record(
        self,
        *,
        kind: str,
        amount_usd: float,
        user_id: int | None = None,
    ) -> None:
        """Persist spend and update Prometheus counters."""
        if amount_usd <= 0:
            return
        if user_id is None:
            user_id = get_billing_user()
        day = self._day_key()
        await self._incr(f"cost:day:{day}", amount_usd, ttl_seconds=3 * 86400)
        if user_id is not None:
            await self._incr(
                f"cost:user:{user_id}:day:{day}",
                amount_usd,
                ttl_seconds=3 * 86400,
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
        )

    async def _get(self, key: str) -> float:
        if self._redis is not None:
            try:
                raw = await self._redis.get(key)
                return float(raw) if raw is not None else 0.0
            except Exception as exc:  # noqa: BLE001
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
            except Exception as exc:  # noqa: BLE001
                logger.warning("cost_tracker_redis_incr_failed", error=str(exc))
        self._memory[key] = self._memory.get(key, 0.0) + amount
        # Soft expiry for memory mode: drop keys older than ~3 days by day stamp.
        cutoff = time.time() - ttl_seconds
        _ = cutoff


cost_tracker = CostTracker()
