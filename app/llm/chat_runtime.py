"""Чат-рантайм: Function Calling loops, SSE deltas, usage accounting."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any, Literal

import structlog

from app.infra.cost_tracker import (
    BudgetExceededError,
    chat_cost_usd,
    cost_tracker,
    isolated_budget_reservation,
)
from app.infra.metrics import (
    note_input_refused,
    note_input_trim,
    note_provider_fallback,
)
from app.llm.context import content_chars
from app.llm.errors import AssistAIError, fallback_reason
from app.llm.models_config import (
    clamp_max_tokens,
    resolve_polza_text_model,
    resolve_text_model,
)
from app.llm.providers import ProviderConfig
from app.llm.tooling import ToolRegistry
from app.llm.transport import _CHAT_PATH, ProviderTransportMixin

logger = structlog.get_logger("AssistAI.chat")


def _as_text(content: Any) -> str:
    """Приводит ``message.content`` к строке. Мультимодальный content сериализуется в JSON."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


async def _execute_tool_isolated(tools: ToolRegistry, call: dict[str, Any]) -> str:
    """Run one tool without inheriting the parent turn's budget reservation."""
    async with isolated_budget_reservation():
        return await tools.execute(call)


def _reasoning_piece(delta: dict[str, Any]) -> str:
    """Достаёт кусок рассуждения. DeepSeek кладёт его в ``reasoning_content``."""
    for key in ("reasoning_content", "reasoning"):
        value = delta.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


# GPT-6: tools на chat/completions только с reasoning_effort=none.
_GPT6_TOOLS_REASONING_NONE: frozenset[str] = frozenset(
    {
        "gpt-6-luna",
        "gpt-6-luna-decisions",
        "gpt-6-sol",
        "gpt-6-terra",
        "gpt-6.1-terra",
    }
)
# GPT-6: tools на chat/completions не поддерживаются (нужен /v1/responses).
# Каталог бота такие модели больше не предлагает; сет оставлен под будущие id.
_GPT6_CHAT_OMIT_TOOLS: frozenset[str] = frozenset()


def _normalize_openai_model_id(model_id: str) -> str:
    name = model_id.strip().casefold()
    if name.startswith("openai/"):
        return name[len("openai/") :]
    return name


def _gpt6_chat_tools_need_reasoning_none(model_id: str) -> bool:
    """True для GPT-6, где tools на chat/completions требуют ``reasoning_effort=none``."""
    return _normalize_openai_model_id(model_id) in _GPT6_TOOLS_REASONING_NONE


def _gpt6_chat_omit_tools(model_id: str) -> bool:
    """True если tools на chat/completions для модели нельзя отправлять."""
    return _normalize_openai_model_id(model_id) in _GPT6_CHAT_OMIT_TOOLS


def _attach_chat_tools(
    payload: dict[str, Any], tools: list[dict[str, Any]], model_id: str
) -> None:
    """Добавляет tools в chat/completions с учётом ограничений GPT-6."""
    if not tools:
        return
    if _gpt6_chat_omit_tools(model_id):
        logger.warning(
            "chat_tools_omitted",
            model=model_id,
            reason="model requires /v1/responses for function tools",
        )
        return
    payload["tools"] = tools
    payload["tool_choice"] = "auto"
    if _gpt6_chat_tools_need_reasoning_none(model_id):
        payload["reasoning_effort"] = "none"


def _apply_generation_options(
    payload: dict[str, Any],
    *,
    response_format: dict[str, Any] | None,
    temperature: float | None,
    max_tokens: int | None,
    reasoning: dict[str, Any] | None = None,
    catalog_model: str | None = None,
) -> None:
    """Добавляет необязательные поля генерации, не трогая tools."""
    if response_format is not None:
        payload["response_format"] = response_format
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        model_id = catalog_model if catalog_model is not None else payload.get("model")
        payload["max_tokens"] = clamp_max_tokens(
            model_id if isinstance(model_id, str) else None,
            max_tokens,
        )
    if reasoning is not None:
        payload["reasoning"] = reasoning


_INPUT_TOO_LARGE = "Запрос слишком большой. Сократите текст или начните новую тему"
_CONTEXT_MARKERS = (
    "context_length_exceeded",
    "maximum context length",
    "too many tokens",
    "context window",
    "input too long",
)


def _needs_input_retry(exc: AssistAIError) -> bool:
    """400 из-за длины входа: один повтор с урезанной историей, без fallback."""
    if exc.status_code not in (400, None):
        return False
    text = (exc.message or "").casefold()
    return any(marker in text for marker in _CONTEXT_MARKERS)


def _needs_max_completion_tokens_retry(exc: AssistAIError) -> bool:
    """True если провайдер отклонил ``max_tokens`` в пользу ``max_completion_tokens``."""
    if exc.status_code not in (400, None):
        return False
    text = (exc.message or "").casefold()
    return "max_completion_tokens" in text and "max_tokens" in text


def _swap_max_tokens_to_completion(payload: dict[str, Any]) -> int | None:
    """Перекладывает ``max_tokens`` → ``max_completion_tokens``. Возвращает лимит."""
    value = payload.pop("max_tokens", None)
    if value is None:
        return None
    payload["max_completion_tokens"] = value
    return int(value)


def _needs_stream_options_retry(exc: AssistAIError) -> bool:
    """True если провайдер отклонил ``stream_options`` или ``include_usage``."""
    if exc.status_code not in (400, None):
        return False
    text = (exc.message or "").casefold()
    return "stream_options" in text or "include_usage" in text


def _adapt_rejected_stream_payload(
    exc: AssistAIError,
    payload: dict[str, Any],
    *,
    allow_drop_stream_options: bool,
    allow_remap_max_tokens: bool,
) -> tuple[bool, bool] | None:
    """Снимает поля, из-за которых стрим ответил 400.

    Возвращает ``(dropped_stream_options, remapped_max_tokens)``
    или ``None``, если повторять запрос нечем.
    """
    dropped = False
    remapped = False
    if (
        allow_drop_stream_options
        and "stream_options" in payload
        and _needs_stream_options_retry(exc)
    ):
        payload.pop("stream_options", None)
        dropped = True
    if (
        allow_remap_max_tokens
        and "max_tokens" in payload
        and _needs_max_completion_tokens_retry(exc)
    ):
        _swap_max_tokens_to_completion(payload)
        remapped = True
    if not dropped and not remapped:
        return None
    return dropped, remapped


_LENGTH_MARK = "…ответ обрезан по лимиту длины"


def _length_mark_suffix(text: str) -> str:
    """Хвост к финальному ответу, если провайдер остановился по ``length``.

    Пустая строка — пометка уже стоит. Иначе одна строка в конце хода.
    """
    if text.rstrip().endswith(_LENGTH_MARK):
        return ""
    if text.endswith("\n") or not text.strip():
        return _LENGTH_MARK
    return f"\n{_LENGTH_MARK}"


def _tool_delta_text(call: dict[str, Any]) -> str:
    """Имя инструмента и короткий preview аргументов для ``ChatDelta``."""
    name = str(call["function"]["name"])
    arguments = str(call["function"].get("arguments") or "")
    preview = arguments if len(arguments) <= 120 else arguments[:117] + "..."
    return f"{name} {preview}".rstrip()


def _assistant_stream_tool_message(
    reasoning: str,
    content: str,
    tool_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    """Сообщение ассистента с tool_calls для следующего круга потока.

    ``reasoning_content`` нужен DeepSeek: без него следующий запрос с tools
    отвечает 400. В чат и в финальный ответ рассуждение не попадает.
    """
    payload: dict[str, Any] = {
        "role": "assistant",
        "tool_calls": tool_calls,
    }
    if reasoning:
        payload["reasoning_content"] = reasoning
    if content:
        payload["content"] = content
    return payload


class _ToolCallBuilder:
    """Склеивает ``delta.tool_calls``, которые в потоке приходят по кускам.

    Имя пишется один раз: шлюз часто повторяет его в каждом дельте.
    Продолжение без ``index`` относится к последнему открытому слоту,
    а не к нулевому — иначе параллельные вызовы склеиваются.
    """

    def __init__(self) -> None:
        self._calls: dict[int, dict[str, Any]] = {}
        self._opened: list[int] = []

    def add(self, delta_call: dict[str, Any]) -> None:
        slot = self._open(self._index_of(delta_call))
        call_id = delta_call.get("id")
        if isinstance(call_id, str) and call_id:
            slot["id"] = call_id
        call_type = delta_call.get("type")
        if isinstance(call_type, str) and call_type:
            slot["type"] = call_type
        function = delta_call.get("function")
        if not isinstance(function, dict):
            return
        name = function.get("name")
        if isinstance(name, str) and name and not slot["function"]["name"]:
            slot["function"]["name"] = name
        arguments = function.get("arguments")
        if isinstance(arguments, str) and arguments:
            slot["function"]["arguments"] += arguments

    def calls(self) -> list[dict[str, Any]]:
        return [self._calls[index] for index in sorted(self._calls)]

    def _index_of(self, delta_call: dict[str, Any]) -> int:
        if "index" not in delta_call or delta_call.get("index") is None:
            return self._opened[-1] if self._opened else 0
        try:
            return int(delta_call["index"])
        except (TypeError, ValueError):
            return self._opened[-1] if self._opened else 0

    def _open(self, index: int) -> dict[str, Any]:
        slot = self._calls.get(index)
        if slot is not None:
            return slot
        slot = {
            "id": "",
            "type": "function",
            "function": {"name": "", "arguments": ""},
        }
        self._calls[index] = slot
        self._opened.append(index)
        return slot


ChatDeltaKind = Literal["content", "tool", "notice"]


@dataclass(frozen=True)
class ChatDelta:
    """Фрагмент потокового ответа ``stream_chat``.

    ``content`` — кусок итогового текста.
    ``tool`` — инструмент сейчас вызывается; в ``text`` имя и аргументы.
    ``notice`` — служебная пометка только для чата. В ``messages`` и в историю
    не пишется, иначе модель увидит её как часть своего ответа.
    Чат успевает показать статус, пока вызов ещё идёт.
    Рассуждения модели в чат не отдаются.
    """

    kind: ChatDeltaKind
    text: str


class ChatRuntimeMixin(ProviderTransportMixin):
    """Миксин chat_completion / stream_chat для ``AssistAIClient``."""

    _tools: ToolRegistry
    get_tools_schema: Callable[[], list[dict[str, Any]]]

    def _catalog_chat_model(self, model: str | None) -> str:
        """Id модели в каталоге бота (после legacy-алиаса)."""
        selected = resolve_text_model(model or self.config.default_text_model) or (
            model or self.config.default_text_model
        )
        return selected

    def _provider_chat_model(self, catalog_model: str, provider: ProviderConfig) -> str:
        """Id модели под конкретного провайдера (Polza может отличаться)."""
        if provider.id == "polza":
            return resolve_polza_text_model(catalog_model) or catalog_model
        return catalog_model

    async def _chat_post_with_fallback(
        self,
        payload: dict[str, Any],
        *,
        catalog_model: str,
        stick_to: ProviderConfig | None = None,
    ) -> tuple[dict[str, Any], ProviderConfig]:
        """Один chat POST: основной провайдер, при недоступности — один fallback."""
        if stick_to is not None:
            req = dict(payload)
            req["model"] = self._provider_chat_model(catalog_model, stick_to)
            return await self._post_json(_CHAT_PATH, req, provider=stick_to), stick_to

        primary, fallback = self._registry.resolve_route("text")
        last_exc: AssistAIError | None = None
        for provider in (primary,) + ((fallback,) if fallback is not None else ()):
            req = dict(payload)
            req["model"] = self._provider_chat_model(catalog_model, provider)
            try:
                data = await self._post_json(_CHAT_PATH, req, provider=provider)
            except AssistAIError as exc:
                last_exc = exc
                if reason := fallback_reason(exc):
                    if reason == "insufficient_balance":
                        logger.error("provider_balance_exhausted", provider=provider.id)
                    if provider is primary and fallback is not None:
                        logger.warning(
                            "provider_fallback",
                            **{
                                "from": primary.id,
                                "to": fallback.id,
                                "reason": reason,
                            },
                        )
                        note_provider_fallback(primary.id, fallback.id, reason)
                        continue
                raise
            return data, provider
        if last_exc is None:
            raise RuntimeError("Обход провайдеров закончился без ошибки")
        raise last_exc

    async def _iter_chat_with_fallback(
        self,
        payload: dict[str, Any],
        *,
        catalog_model: str,
        stick_to: ProviderConfig | None = None,
    ) -> AsyncIterator[tuple[dict[str, Any], ProviderConfig]]:
        """SSE chat: основной провайдер; fallback только до первого chunk."""
        if stick_to is not None:
            req = dict(payload)
            req["model"] = self._provider_chat_model(catalog_model, stick_to)
            async for chunk in self._iter_chat_chunks(req, provider=stick_to):
                yield chunk, stick_to
            return

        primary, fallback = self._registry.resolve_route("text")
        providers: tuple[ProviderConfig, ...] = (primary,) + (
            (fallback,) if fallback is not None else ()
        )
        last_exc: AssistAIError | None = None
        for provider in providers:
            req = dict(payload)
            req["model"] = self._provider_chat_model(catalog_model, provider)
            yielded = False
            try:
                async for chunk in self._iter_chat_chunks(req, provider=provider):
                    yielded = True
                    yield chunk, provider
                return
            except AssistAIError as exc:
                last_exc = exc
                if yielded:
                    raise
                reason = fallback_reason(exc)
                if reason is None or provider is not primary or fallback is None:
                    raise
                if reason == "insufficient_balance":
                    logger.error("provider_balance_exhausted", provider=provider.id)
                logger.warning(
                    "provider_fallback",
                    **{"from": primary.id, "to": fallback.id, "reason": reason},
                )
                note_provider_fallback(primary.id, fallback.id, reason)
        if last_exc is None:
            raise RuntimeError("Обход провайдеров закончился без ошибки")
        raise last_exc

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        *,
        use_tools: bool = True,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning: dict[str, Any] | None = None,
    ) -> str:
        """Ведёт диалог с моделью и сама выполняет запрошенные инструменты.

        В каждый запрос уходит поле ``tools``. Пока ``finish_reason`` равен
        ``tool_calls``, метод разбирает ``function.arguments`` как JSON-строку,
        вызывает ``execute`` нужного инструмента и добавляет в ``messages``
        ответ ассистента и сообщение ``role: "tool"`` с тем же ``tool_call_id``.
        Цикл останавливается на обычном текстовом ответе и возвращает его ``content``.

        Больше ``max_tool_iterations`` кругов не выполняется: это защита от
        модели, которая запрашивает инструменты без остановки. В этом случае
        выбрасывается ``AssistAIError``, уже полученные результаты остаются в ``messages``.

        Args:
            messages: История диалога в формате OpenAI. Метод дополняет её на месте.
            model: Идентификатор модели ProxyAPI. По умолчанию ``default_text_model``.

        Returns:
            Итоговый текст ответа модели.

        Raises:
            ValueError: Список сообщений пуст.
            AssistAIError: Ошибка HTTP, неразборчивый ответ или лимит итераций.
        """
        if not messages:
            raise ValueError("Список messages не может быть пустым.")

        catalog_model = self._catalog_chat_model(model)
        tools = self.get_tools_schema()
        active_provider: ProviderConfig | None = None
        input_anchor = self._input_anchor(messages)

        for iteration in range(1, self.config.max_tool_iterations + 1):
            input_retried = False
            if iteration > 1:
                await self._ensure_round_budget(messages)
            selected_model = self._provider_chat_model(
                catalog_model, active_provider or self._endpoint("text")
            )
            payload: dict[str, Any] = {
                "model": selected_model,
                "messages": messages,
            }
            if use_tools:
                _attach_chat_tools(payload, tools, selected_model)
                if tool_choice is not None:
                    payload["tool_choice"] = tool_choice
            if iteration > 1:
                payload["messages"] = self._messages_for_provider(
                    messages,
                    protect_tail=self._turn_protect_tail(messages, input_anchor),
                    token_limit=None,
                    stage="round",
                    model=catalog_model,
                )
            _apply_generation_options(
                payload,
                response_format=response_format,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning=reasoning,
                catalog_model=catalog_model,
            )

            logger.info(
                "chat_completion: модель=%s, итерация=%s/%s, сообщений=%s",
                selected_model,
                iteration,
                self.config.max_tool_iterations,
                len(messages),
            )
            try:
                data, active_provider = await self._chat_post_with_fallback(
                    payload,
                    catalog_model=catalog_model,
                    stick_to=active_provider,
                )
            except AssistAIError as exc:
                if not input_retried and _needs_input_retry(exc):
                    input_retried = True
                    payload["messages"] = self._messages_for_provider(
                        messages,
                        protect_tail=self._turn_protect_tail(messages, input_anchor),
                        token_limit=self._half_input_budget(),
                        stage="reactive",
                        model=catalog_model,
                    )
                    logger.info("chat_input_retry", model=catalog_model)
                    data, active_provider = await self._chat_post_with_fallback(
                        payload,
                        catalog_model=catalog_model,
                        stick_to=active_provider,
                    )
                elif "max_tokens" in payload and _needs_max_completion_tokens_retry(
                    exc
                ):
                    swapped = _swap_max_tokens_to_completion(payload)
                    logger.info(
                        "chat_max_tokens_remapped",
                        model=selected_model,
                        max_completion_tokens=swapped,
                    )
                    data, active_provider = await self._chat_post_with_fallback(
                        payload,
                        catalog_model=catalog_model,
                        stick_to=active_provider,
                    )
                else:
                    raise
            message, finish_reason = self._extract_choice(data)
            await self._record_chat_usage(
                data,
                messages,
                assistant_message=message,
                provider_id=active_provider.id,
                provider=active_provider,
            )
            raw_tool_calls = message.get("tool_calls") or []
            tool_calls = [
                self._tools.normalize_tool_call(call)
                for call in raw_tool_calls
                if isinstance(call, dict)
            ]

            # Модель иногда прикладывает tool_calls при другом finish_reason.
            # Такой ответ тоже нужно выполнить, иначе вызов потеряется.
            if use_tools and (finish_reason == "tool_calls" or tool_calls):
                if not tool_calls:
                    raise AssistAIError(
                        "Модель вернула finish_reason=tool_calls без списка tool_calls."
                    )
                names = ", ".join(call["function"]["name"] for call in tool_calls)
                logger.info("Модель запросила инструменты: %s", names)
                messages.append(self._tools.assistant_tool_message(message, tool_calls))
                results = await asyncio.gather(
                    *(_execute_tool_isolated(self._tools, call) for call in tool_calls)
                )
                for call, result in zip(tool_calls, results, strict=True):
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": result,
                        }
                    )
                if iteration >= self.config.max_tool_iterations:
                    logger.error(
                        "Достигнут лимит итераций инструментов: %s",
                        self.config.max_tool_iterations,
                    )
                    raise AssistAIError(
                        "Превышен лимит вызовов инструментов: "
                        f"{self.config.max_tool_iterations}"
                    )
                continue

            break

        content = _as_text(message.get("content"))
        messages.append({"role": "assistant", "content": content})
        logger.info(
            "chat_completion: финальный ответ, finish_reason=%s, символов=%s",
            finish_reason,
            len(content),
        )
        return content

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        *,
        use_tools: bool = True,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[ChatDelta]:
        """Потоковый диалог с теми же инструментами, что и ``chat_completion``.

        В запрос добавляется ``stream: true`` и
        ``stream_options: {"include_usage": true}``: без флага OpenAI-совместимый
        API не кладёт usage в последний чанк, и раунд считается эвристикой.
        Если провайдер отвечает 400 на ``stream_options``, поле снимается
        и запрос повторяется. Наружу выходят только текст ответа и отметки
        об инструментах. Куски рассуждения в поток и в финальное сообщение
        ассистента не попадают.

        Вызовы инструментов собираются из ``delta.tool_calls``. Отметка
        ``tool`` уходит до выполнения, чтобы чат показал статус, пока
        инструмент работает. Для DeepSeek рассуждение сохраняется только
        во внутренних tool-кругах как ``reasoning_content``: без него
        следующий запрос с tools отвечает 400.

        Таймаут чтения не короче 180 секунд. Соединение ограничено ``config.timeout``.

        Args:
            messages: История диалога. Метод дополняет её на месте.
            model: Модель ProxyAPI. По умолчанию ``default_text_model``.

        Yields:
            Фрагменты ответа и отметки о выполненных инструментах.

        Raises:
            ValueError: Список сообщений пуст.
            AssistAIError: Ошибка HTTP, обрыв потока или лимит итераций.
        """
        if not messages:
            raise ValueError("Список messages не может быть пустым.")

        catalog_model = self._catalog_chat_model(model)
        tools = self.get_tools_schema()
        active_provider: ProviderConfig | None = None
        input_anchor = self._input_anchor(messages)

        for iteration in range(1, self.config.max_tool_iterations + 1):
            input_retried = False
            if iteration > 1:
                await self._ensure_round_budget(messages)
            selected_model = self._provider_chat_model(
                catalog_model, active_provider or self._endpoint("text")
            )
            payload: dict[str, Any] = {
                "model": selected_model,
                "messages": messages,
                "stream": True,
                # ProxyAPI/OpenAI отдают usage в стриме только с этим флагом.
                # Polza часто шлёт его и без флага; отказ 400 снимает поле.
                "stream_options": {"include_usage": True},
            }
            if use_tools:
                _attach_chat_tools(payload, tools, selected_model)
                if tool_choice is not None:
                    payload["tool_choice"] = tool_choice
            if iteration > 1:
                payload["messages"] = self._messages_for_provider(
                    messages,
                    protect_tail=self._turn_protect_tail(messages, input_anchor),
                    token_limit=None,
                    stage="round",
                    model=catalog_model,
                )
            _apply_generation_options(
                payload,
                response_format=response_format,
                temperature=temperature,
                max_tokens=max_tokens,
                catalog_model=catalog_model,
            )

            logger.info(
                "stream_chat: модель=%s, итерация=%s/%s, сообщений=%s",
                selected_model,
                iteration,
                self.config.max_tool_iterations,
                len(messages),
            )
            reasoning_parts: list[str] = []
            content_parts: list[str] = []
            tool_builder = _ToolCallBuilder()
            finish_reason = ""
            dropped_stream_options = False
            remapped_max_tokens = False
            saw_chunk = False
            last_usage: dict[str, Any] | None = None
            billed_model: str | None = None

            while True:
                try:
                    async for chunk, active_provider in self._iter_chat_with_fallback(
                        payload,
                        catalog_model=catalog_model,
                        stick_to=active_provider,
                    ):
                        saw_chunk = True
                        raw_usage = chunk.get("usage")
                        if isinstance(raw_usage, dict):
                            last_usage = raw_usage
                        raw_model = chunk.get("model")
                        if isinstance(raw_model, str) and raw_model:
                            billed_model = raw_model
                        choices = chunk.get("choices")
                        if not isinstance(choices, list) or not choices:
                            continue
                        choice = choices[0]
                        if not isinstance(choice, dict):
                            continue
                        reason = choice.get("finish_reason")
                        if isinstance(reason, str) and reason:
                            finish_reason = reason
                        delta = choice.get("delta") or {}
                        if not isinstance(delta, dict):
                            continue

                        reasoning = _reasoning_piece(delta)
                        if reasoning:
                            reasoning_parts.append(reasoning)

                        content = delta.get("content")
                        if isinstance(content, str) and content:
                            content_parts.append(content)
                            yield ChatDelta("content", content)

                        raw_calls = delta.get("tool_calls")
                        if isinstance(raw_calls, list):
                            for call in raw_calls:
                                if isinstance(call, dict):
                                    tool_builder.add(call)
                    break
                except AssistAIError as exc:
                    # Куски уже ушли в чат: повтор продублирует текст.
                    if saw_chunk:
                        raise
                    adapted = _adapt_rejected_stream_payload(
                        exc,
                        payload,
                        allow_drop_stream_options=not dropped_stream_options,
                        allow_remap_max_tokens=not remapped_max_tokens,
                    )
                    if adapted is None:
                        if saw_chunk or input_retried or not _needs_input_retry(exc):
                            raise
                        input_retried = True
                        payload["messages"] = self._messages_for_provider(
                            messages,
                            protect_tail=self._turn_protect_tail(
                                messages, input_anchor
                            ),
                            token_limit=self._half_input_budget(),
                            stage="reactive",
                            model=catalog_model,
                        )
                        logger.info("chat_input_retry", model=catalog_model)
                        reasoning_parts.clear()
                        content_parts.clear()
                        tool_builder = _ToolCallBuilder()
                        finish_reason = ""
                        last_usage = None
                        billed_model = None
                        saw_chunk = False
                        continue
                    dropped_now, remapped_now = adapted
                    if dropped_now:
                        dropped_stream_options = True
                        logger.info(
                            "chat_stream_options_dropped",
                            model=selected_model,
                        )
                    if remapped_now:
                        remapped_max_tokens = True
                        logger.info(
                            "chat_max_tokens_remapped",
                            model=selected_model,
                            max_completion_tokens=payload.get("max_completion_tokens"),
                            stream=True,
                        )
                    reasoning_parts.clear()
                    content_parts.clear()
                    tool_builder = _ToolCallBuilder()
                    finish_reason = ""
                    last_usage = None
                    billed_model = None
                    saw_chunk = False

            tool_calls = [
                self._tools.normalize_tool_call(call) for call in tool_builder.calls()
            ]
            reasoning_text = "".join(reasoning_parts)
            content_text = "".join(content_parts)
            wants_tools = (
                use_tools
                and (finish_reason == "tool_calls" or bool(tool_calls))
                and bool(tool_calls)
            )
            if use_tools and finish_reason == "tool_calls" and not tool_calls:
                raise AssistAIError(
                    "Модель вернула finish_reason=tool_calls без списка tool_calls."
                )
            if wants_tools:
                names = ", ".join(call["function"]["name"] for call in tool_calls)
                logger.info("Модель запросила инструменты: %s", names)
                messages.append(
                    _assistant_stream_tool_message(
                        reasoning_text, content_text, tool_calls
                    )
                )
                # Один record на раунд: факт usage, иначе эвристика.
                await self._bill_stream_round(
                    messages,
                    content_text
                    or " ".join(
                        str(call["function"].get("arguments") or "")
                        for call in tool_calls
                    ),
                    usage=last_usage,
                    model=billed_model or selected_model,
                    provider=active_provider,
                )
                # До await: потребитель успевает сменить «Готовлю ответ…»,
                # пока generate_image / generate_video ещё не стартовали.
                for call in tool_calls:
                    yield ChatDelta("tool", _tool_delta_text(call))
                results = await asyncio.gather(
                    *(_execute_tool_isolated(self._tools, call) for call in tool_calls)
                )
                for call, result in zip(tool_calls, results, strict=True):
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": result,
                        }
                    )
                if iteration >= self.config.max_tool_iterations:
                    logger.error(
                        "Достигнут лимит итераций инструментов: %s",
                        self.config.max_tool_iterations,
                    )
                    raise AssistAIError(
                        "Превышен лимит вызовов инструментов: "
                        f"{self.config.max_tool_iterations}"
                    )
                continue

            messages.append({"role": "assistant", "content": content_text})
            await self._bill_stream_round(
                messages,
                content_text,
                usage=last_usage,
                model=billed_model or selected_model,
                provider=active_provider,
            )
            if finish_reason == "length":
                suffix = _length_mark_suffix(content_text)
                if suffix:
                    yield ChatDelta("notice", suffix)
            logger.info(
                "stream_chat: финальный ответ, finish_reason=%s, символов=%s",
                finish_reason,
                len(content_text),
            )
            return

        raise AssistAIError(
            f"Превышен лимит вызовов инструментов: {self.config.max_tool_iterations}"
        )

    @staticmethod
    def _input_anchor(messages: list[dict[str, Any]]) -> int:
        """Индекс текущего user: всё после него — этот ход, его не режем."""
        if messages and messages[-1].get("role") == "user":
            return len(messages) - 1
        return len(messages)

    @staticmethod
    def _turn_protect_tail(messages: list[dict[str, Any]], anchor: int) -> int:
        return max(0, len(messages) - anchor)

    @staticmethod
    def _half_input_budget() -> int:
        from app.infra.settings import get_settings

        return max(1, get_settings().turn_input_token_limit // 2)

    def _messages_for_provider(
        self,
        messages: list[dict[str, Any]],
        *,
        protect_tail: int,
        token_limit: int | None,
        stage: str,
        model: str,
    ) -> list[dict[str, Any]]:
        """Копия входа для провайдера. Общий список хода не сдвигается."""
        from app.infra.settings import get_settings
        from app.llm.context import estimate_tokens
        from app.storage.message_trim import shrink_to_token_limit, trim_messages

        settings = get_settings()
        limit = settings.turn_input_token_limit if token_limit is None else token_limit
        before = len(messages)
        fitted = trim_messages(
            messages,
            max(before, 1),
            max_chars=settings.history_max_chars,
            protect_tail=protect_tail,
        )
        fitted = shrink_to_token_limit(
            fitted,
            token_limit=limit,
            chars_per_token=settings.chars_per_token,
            protect_tail=protect_tail,
        )
        tokens = estimate_tokens(fitted, chars_per_token=settings.chars_per_token)
        if tokens > limit:
            note_input_refused("too_large")
            raise AssistAIError(_INPUT_TOO_LARGE, error_code="INPUT_TOO_LARGE")
        dropped = before - len(fitted)
        if dropped > 0:
            note_input_trim(stage)
            logger.info(
                "input_trimmed",
                model=model,
                kept=len(fitted),
                dropped=dropped,
                tokens=tokens,
            )
        return fitted

    async def _ensure_round_budget(self, messages: list[dict[str, Any]]) -> None:
        """Reserve headroom for the next tool-loop round after the hold is burned.

        Handlers reserve once for the turn; ``record`` only burns
        ``min(reserved, actual)``. Later rounds may need a fresh hold so the
        daily cap is checked again before another paid LLM call.
        """
        input_chars = sum(content_chars(m.get("content")) for m in messages)
        estimated = cost_tracker.estimate_chat_cost(
            input_chars=max(input_chars, 200), output_chars=400
        )
        try:
            await cost_tracker.check_budget(estimated=estimated)
        except BudgetExceededError:
            raise

    async def _bill_stream_round(
        self,
        messages: list[dict[str, Any]],
        output_text: str,
        *,
        usage: dict[str, Any] | None,
        model: str | None,
        provider: ProviderConfig | None,
    ) -> None:
        """Один record на раунд stream: финальный usage-chunk или эвристика."""
        provider_id = provider.id if provider is not None else None
        if usage is None:
            await self._record_chat_estimate(
                messages,
                output_text,
                provider_id=provider_id,
                provider=provider,
            )
            return
        await self._record_chat_usage(
            {"usage": usage, "model": model},
            messages[:-1],
            assistant_message=messages[-1] if messages else None,
            provider_id=provider_id,
            provider=provider,
        )

    async def _record_chat_usage(
        self,
        data: dict[str, Any],
        messages: list[dict[str, Any]],
        *,
        assistant_message: dict[str, Any] | None = None,
        provider_id: str | None = None,
        provider: ProviderConfig | None = None,
    ) -> None:
        """Record spend from ``cost_rub``, token usage, or a chars heuristic."""
        usage = data.get("usage") if isinstance(data, dict) else None
        if not isinstance(usage, dict):
            usage = None
        prompt_tokens = None
        completion_tokens = None
        if usage is not None:
            raw_in = usage.get("prompt_tokens")
            raw_out = usage.get("completion_tokens")
            if isinstance(raw_in, int) and not isinstance(raw_in, bool):
                prompt_tokens = raw_in
            if isinstance(raw_out, int) and not isinstance(raw_out, bool):
                completion_tokens = raw_out
        input_chars = sum(len(_as_text(m.get("content"))) for m in messages)
        output_chars = 0
        if isinstance(assistant_message, dict):
            output_chars = len(_as_text(assistant_message.get("content")))
            for call in assistant_message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                function = call.get("function")
                if isinstance(function, dict):
                    output_chars += len(str(function.get("arguments") or ""))
        amount, source, rub = chat_cost_usd(
            usage=usage,
            provider=provider,
            rub_per_usd=self.config.rub_per_usd,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            input_chars=input_chars,
            output_chars=output_chars,
        )
        model_name = data.get("model") if isinstance(data, dict) else None
        await cost_tracker.record(
            kind="chat",
            amount_usd=amount,
            provider=provider_id or self._endpoint("text").id,
            model=model_name if isinstance(model_name, str) else None,
            source=source,
            cost_rub=rub,
        )

    async def _record_chat_estimate(
        self,
        messages: list[dict[str, Any]],
        output_text: str,
        *,
        provider_id: str | None = None,
        provider: ProviderConfig | None = None,
    ) -> None:
        """Estimate stream_chat spend when usage chunks are unavailable."""
        input_chars = sum(content_chars(m.get("content")) for m in messages[:-1])
        amount, source, rub = chat_cost_usd(
            usage=None,
            provider=provider,
            rub_per_usd=self.config.rub_per_usd,
            input_chars=input_chars,
            output_chars=len(output_text),
        )
        await cost_tracker.record(
            kind="chat",
            amount_usd=amount,
            provider=provider_id or self._endpoint("text").id,
            source=source,
            cost_rub=rub,
        )

    @staticmethod
    def _extract_choice(data: dict[str, Any]) -> tuple[dict[str, Any], str]:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise AssistAIError("ProxyAPI не вернул choices.")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise AssistAIError("ProxyAPI вернул некорректный элемент choices.")
        message = choice.get("message") or {}
        if not isinstance(message, dict):
            raise AssistAIError("ProxyAPI вернул некорректное сообщение модели.")
        finish_reason = choice.get("finish_reason")
        return message, finish_reason if isinstance(finish_reason, str) else ""
