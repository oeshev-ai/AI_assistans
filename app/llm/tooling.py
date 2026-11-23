"""Базовый контракт Function Calling и реестр инструментов."""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from typing import Any, Protocol

import structlog

logger = structlog.get_logger("AssistAI.tooling")


def error_json(message: str) -> str:
    """JSON-строка, которую можно безопасно вернуть модели вместо падения диалога."""
    return json.dumps({"error": message}, ensure_ascii=False)


def exception_text(exc: Exception) -> str:
    """Safe short error for model/tool JSON — strip paths and URLs."""
    text = str(exc).strip() or exc.__class__.__name__
    # Avoid leaking local paths / internal URLs back into the model context.
    if "://" in text or "\\" in text or text.count("/") >= 3:
        return f"{exc.__class__.__name__}: internal error"
    if len(text) > 240:
        return text[:237] + "..."
    return text


# Совместимость со старыми импортами ``_error_json`` / ``_exception_text``.
_error_json = error_json
_exception_text = exception_text


class BaseTool(ABC):
    """Инструмент, который модель может запросить через Function Calling.

    ``execute`` всегда возвращает JSON-строку. Исключение внутри инструмента
    не должно выходить наружу: его текст кладётся в поле ``error``.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Имя функции, которое модель указывает в ``tool_calls``."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Когда вызывать функцию. По этому тексту модель выбирает инструмент."""

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON Schema аргументов в формате OpenAI tools."""

    def to_openai_format(self) -> dict[str, Any]:
        """Собирает описание инструмента в объект поля ``tools``."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str:
        """Выполняет инструмент и возвращает JSON-строку.

        При любой ошибке реализация возвращает ``{"error": "..."}``,
        а не пробрасывает исключение.
        """


class ToolHost(Protocol):
    """Минимальный контракт клиента для media/image tools."""

    def register_tool(self, tool: BaseTool) -> None: ...


class ToolRegistry:
    """Реестр инструментов и диспетчер вызовов Function Calling."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        if not tool.name:
            raise ValueError("У инструмента должно быть непустое имя.")
        self._tools[tool.name] = tool
        logger.info("Зарегистрирован инструмент %s", tool.name)

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def schema(self) -> list[dict[str, Any]]:
        return [tool.to_openai_format() for tool in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    @staticmethod
    def parse_arguments(raw: Any) -> dict[str, Any]:
        """Разбирает ``function.arguments``. ProxyAPI передаёт их строкой JSON."""
        if raw is None or raw == "":
            return {}
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError("arguments должны быть JSON-объектом.")
            return parsed
        raise ValueError("Неподдерживаемый тип arguments.")

    @staticmethod
    def normalize_tool_call(call: dict[str, Any]) -> dict[str, Any]:
        """Приводит вызов к виду, который ProxyAPI принимает в следующем запросе.

        ``function.arguments`` в ответе модели — строка JSON. Если провайдер
        уже отдал объект, он снова сериализуется в строку.
        """
        function = call.get("function")
        if not isinstance(function, dict):
            function = {}

        raw_arguments = function.get("arguments", "{}")
        if isinstance(raw_arguments, str):
            arguments = raw_arguments
        else:
            arguments = json.dumps(raw_arguments, ensure_ascii=False)

        call_id = call.get("id")
        if not isinstance(call_id, str) or not call_id:
            call_id = f"call_{uuid.uuid4().hex}"

        name = function.get("name")
        return {
            "id": call_id,
            "type": "function",
            "function": {
                "name": name if isinstance(name, str) else "",
                "arguments": arguments,
            },
        }

    @staticmethod
    def assistant_tool_message(
        message: dict[str, Any],
        tool_calls: list[dict[str, Any]],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "role": "assistant",
            "tool_calls": tool_calls,
        }
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            payload["content"] = content
        # DeepSeek thinking + tools: без reasoning_content следующий запрос даёт 400.
        reasoning = message.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            payload["reasoning_content"] = reasoning
        else:
            reasoning_alt = message.get("reasoning")
            if isinstance(reasoning_alt, str) and reasoning_alt:
                payload["reasoning_content"] = reasoning_alt
        return payload

    async def execute(self, call: dict[str, Any]) -> str:
        """Выполняет один tool call. Любой сбой возвращается модели как JSON-ошибка."""
        function = call.get("function")
        if not isinstance(function, dict):
            return error_json("В вызове инструмента нет объекта function.")

        name = function.get("name")
        if not isinstance(name, str) or not name:
            return error_json("Модель не указала имя инструмента.")

        try:
            arguments = self.parse_arguments(function.get("arguments"))
        except (ValueError, json.JSONDecodeError) as exc:
            logger.error("Не удалось разобрать аргументы %s: %s", name, exc)
            return error_json(
                f"Аргументы инструмента {name} не являются JSON-объектом."
            )

        tool = self._tools.get(name)
        if tool is None:
            logger.error("Неизвестный инструмент: %s", name)
            return error_json(f"Неизвестный инструмент: {name}")

        # Keys only — prompts/PII must not land in info logs.
        arg_keys = sorted(arguments) if isinstance(arguments, dict) else []
        logger.info("Вызов инструмента %s, arg_keys=%s", name, arg_keys)
        try:
            result = await tool.execute(**arguments)
        except Exception as exc:
            # Запасной контур: инструмент обязан гасить ошибки сам, но диалог
            # не прерывается, даже если реализация пробросила исключение.
            logger.exception("Инструмент %s пробросил исключение", name)
            return error_json(exception_text(exc))

        if not isinstance(result, str):
            result = json.dumps(result, ensure_ascii=False, default=str)
        if not result:
            logger.error("Инструмент %s вернул пустой ответ", name)
            return error_json(f"Инструмент {name} вернул пустой ответ.")

        logger.info("Инструмент %s вернул %s символов", name, len(result))
        return result
