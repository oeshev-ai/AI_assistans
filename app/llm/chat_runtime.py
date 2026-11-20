"""Чат-рантайм: Function Calling loops, SSE deltas, usage accounting."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any, Literal

import structlog

from app.infra.cost_tracker import cost_tracker, isolated_budget_reservation
from app.llm.errors import AssistAIError
from app.llm.models_config import resolve_polza_text_model, resolve_text_model
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
) -> None:
    """Добавляет необязательные поля генерации, не трогая tools."""
    if response_format is not None:
        payload["response_format"] = response_format
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens


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
    """Склеивает ``delta.tool_calls``, которые в потоке приходят по кускам."""

    def __init__(self) -> None:
        self._calls: dict[int, dict[str, Any]] = {}

    def add(self, delta_call: dict[str, Any]) -> None:
        raw_index = delta_call.get("index", 0)
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            index = 0
        slot = self._calls.setdefault(
            index,
            {
                "id": "",
                "type": "function",
                "function": {"name": "", "arguments": ""},
            },
        )
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
        if isinstance(name, str) and name:
            slot["function"]["name"] += name
        arguments = function.get("arguments")
        if isinstance(arguments, str) and arguments:
            slot["function"]["arguments"] += arguments

    def calls(self) -> list[dict[str, Any]]:
        return [self._calls[index] for index in sorted(self._calls)]


ChatDeltaKind = Literal["content", "tool"]


@dataclass(frozen=True)
class ChatDelta:
    """Фрагмент потокового ответа ``stream_chat``.

    ``content`` — кусок итогового текста.
    ``tool`` — инструмент уже выполнен, в ``text`` его имя и аргументы.
    Рассуждения модели в чат не отдаются.
    """

    kind: ChatDeltaKind
    text: str


class ChatRuntimeMixin(ProviderTransportMixin):
    """Миксин chat_completion / stream_chat для ``AssistAIClient``."""

    _tools: ToolRegistry
    get_tools_schema: Callable[[], list[dict[str, Any]]]

    def _resolve_chat_model(self, model: str | None) -> str:
        """Модель для chat: legacy-алиас, на Polza — ещё алиас каталога."""
        selected = resolve_text_model(model or self.config.default_text_model) or (
            model or self.config.default_text_model
        )
        if self._endpoint("text").id == "polza":
            return resolve_polza_text_model(selected) or selected
        return selected

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

        selected_model = self._resolve_chat_model(model)
        tools = self.get_tools_schema()

        for iteration in range(1, self.config.max_tool_iterations + 1):
            payload: dict[str, Any] = {
                "model": selected_model,
                "messages": messages,
            }
            if use_tools:
                _attach_chat_tools(payload, tools, selected_model)
                if tool_choice is not None:
                    payload["tool_choice"] = tool_choice
            _apply_generation_options(
                payload,
                response_format=response_format,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            logger.info(
                "chat_completion: модель=%s, итерация=%s/%s, сообщений=%s",
                selected_model,
                iteration,
                self.config.max_tool_iterations,
                len(messages),
            )
            try:
                data = await self._post_json(_CHAT_PATH, payload)
            except AssistAIError as exc:
                if (
                    "max_tokens" in payload
                    and _needs_max_completion_tokens_retry(exc)
                ):
                    swapped = _swap_max_tokens_to_completion(payload)
                    logger.info(
                        "chat_max_tokens_remapped",
                        model=selected_model,
                        max_completion_tokens=swapped,
                    )
                    data = await self._post_json(_CHAT_PATH, payload)
                else:
                    raise
            message, finish_reason = self._extract_choice(data)
            await self._record_chat_usage(data, messages, assistant_message=message)
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

        В запрос добавляется ``stream: true``. Наружу выходят только текст
        ответа и отметки об инструментах. Куски рассуждения в поток и в
        финальное сообщение ассистента не попадают.

        Вызовы инструментов собираются из ``delta.tool_calls`` и выполняются
        после закрытия потока. Для DeepSeek рассуждение сохраняется только
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

        selected_model = self._resolve_chat_model(model)
        tools = self.get_tools_schema()

        for iteration in range(1, self.config.max_tool_iterations + 1):
            payload: dict[str, Any] = {
                "model": selected_model,
                "messages": messages,
                "stream": True,
            }
            if use_tools:
                _attach_chat_tools(payload, tools, selected_model)
                if tool_choice is not None:
                    payload["tool_choice"] = tool_choice
            _apply_generation_options(
                payload,
                response_format=response_format,
                temperature=temperature,
                max_tokens=max_tokens,
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
            remapped_max_tokens = False

            while True:
                try:
                    async for chunk in self._iter_chat_chunks(payload):
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
                    if (
                        remapped_max_tokens
                        or "max_tokens" not in payload
                        or not _needs_max_completion_tokens_retry(exc)
                    ):
                        raise
                    remapped_max_tokens = True
                    swapped = _swap_max_tokens_to_completion(payload)
                    logger.info(
                        "chat_max_tokens_remapped",
                        model=selected_model,
                        max_completion_tokens=swapped,
                        stream=True,
                    )
                    reasoning_parts.clear()
                    content_parts.clear()
                    tool_builder = _ToolCallBuilder()
                    finish_reason = ""

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
                # Charge the tool-calling round; final answer is billed separately.
                await self._record_chat_estimate(
                    messages,
                    content_text
                    or " ".join(
                        str(call["function"].get("arguments") or "")
                        for call in tool_calls
                    ),
                )
                results = await asyncio.gather(
                    *(_execute_tool_isolated(self._tools, call) for call in tool_calls)
                )
                for call, result in zip(tool_calls, results, strict=True):
                    name = str(call["function"]["name"])
                    arguments = str(call["function"]["arguments"])
                    preview = (
                        arguments if len(arguments) <= 120 else arguments[:117] + "..."
                    )
                    yield ChatDelta("tool", f"{name} {preview}".rstrip())
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
            await self._record_chat_estimate(messages, content_text)
            logger.info(
                "stream_chat: финальный ответ, finish_reason=%s, символов=%s",
                finish_reason,
                len(content_text),
            )
            return

        raise AssistAIError(
            f"Превышен лимит вызовов инструментов: {self.config.max_tool_iterations}"
        )

    async def _record_chat_usage(
        self,
        data: dict[str, Any],
        messages: list[dict[str, Any]],
        *,
        assistant_message: dict[str, Any] | None = None,
    ) -> None:
        """Record spend from provider ``usage`` or a chars heuristic."""
        usage = data.get("usage") if isinstance(data, dict) else None
        prompt_tokens = None
        completion_tokens = None
        if isinstance(usage, dict):
            raw_in = usage.get("prompt_tokens")
            raw_out = usage.get("completion_tokens")
            if isinstance(raw_in, int):
                prompt_tokens = raw_in
            if isinstance(raw_out, int):
                completion_tokens = raw_out
        input_chars = sum(len(_as_text(m.get("content"))) for m in messages)
        output_chars = 0
        if completion_tokens is None and isinstance(assistant_message, dict):
            output_chars = len(_as_text(assistant_message.get("content")))
            for call in assistant_message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                function = call.get("function")
                if isinstance(function, dict):
                    output_chars += len(str(function.get("arguments") or ""))
        amount = cost_tracker.estimate_chat_cost(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            input_chars=0 if prompt_tokens is not None else input_chars,
            output_chars=0 if completion_tokens is not None else output_chars,
        )
        model_name = data.get("model") if isinstance(data, dict) else None
        await cost_tracker.record(
            kind="chat",
            amount_usd=amount,
            provider=self._endpoint("text").id,
            model=model_name if isinstance(model_name, str) else None,
            source="usage" if prompt_tokens is not None else "flat",
        )

    async def _record_chat_estimate(
        self, messages: list[dict[str, Any]], output_text: str
    ) -> None:
        """Estimate stream_chat spend when usage chunks are unavailable."""
        input_chars = sum(len(_as_text(m.get("content"))) for m in messages[:-1])
        amount = cost_tracker.estimate_chat_cost(
            input_chars=input_chars,
            output_chars=len(output_text),
        )
        await cost_tracker.record(
            kind="chat",
            amount_usd=amount,
            provider=self._endpoint("text").id,
            source="flat",
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
