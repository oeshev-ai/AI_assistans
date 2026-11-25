"""Асинхронный клиент LLM для Telegram-бота.

``AssistAIClient`` ходит в ProxyAPI (совместимый с OpenAI формат):
текстовые ответы с Function Calling и генерация изображений.
Погода и конвертация валют подключены как инструменты, которые модель
запрашивает сама, а выполняет этот модуль.

Реализация разнесена по доменам: ``config``, ``tooling``, ``builtin_tools``,
``transport``, ``chat_runtime``, ``image_ops``, ``media_api``.
"""

from __future__ import annotations

from typing import Any

import httpx

from app.llm.builtin_tools import register_data_tools
from app.llm.chat_runtime import ChatDelta, ChatRuntimeMixin
from app.llm.config import AssistAIConfig
from app.llm.errors import AssistAIError
from app.llm.image_ops import (
    GenerateImageTool,
    ImageModelScope,
    ImageOpsMixin,
    image_model_family,
    max_edit_images,
    note_generation_id,
    prepare_edit_files,
    supports_multi_image,
)
from app.llm.media_api import (
    MediaApiMixin,
    SpeechResult,
    VideoResult,
    resolve_video_resource_url,
)
from app.llm.openweather import OpenWeatherClient
from app.llm.providers import ProviderRegistry
from app.llm.tooling import BaseTool, ToolRegistry

__all__ = [
    "AssistAIClient",
    "AssistAIConfig",
    "AssistAIError",
    "BaseTool",
    "ChatDelta",
    "ImageModelScope",
    "SpeechResult",
    "VideoResult",
    "image_model_family",
    "max_edit_images",
    "note_generation_id",
    "prepare_edit_files",
    "resolve_video_resource_url",
    "supports_multi_image",
]


class AssistAIClient(
    ChatRuntimeMixin,
    ImageOpsMixin,
    MediaApiMixin,
):
    """Единый асинхронный клиент ProxyAPI: чат с инструментами и картинки.

    Экземпляр владеет ``httpx.AsyncClient`` и, если задан ключ OpenWeather,
    собственным ``OpenWeatherClient``. Закрывайте их через ``close``
    или ``async with``.

    ``chat_completion`` дописывает реплики ассистента и результаты инструментов
    в переданный список ``messages``. Один и тот же список не стоит обрабатывать
    двумя вызовами одновременно. Сам HTTP-клиент такие вызовы переносит,
    если у каждого свой список сообщений.
    """

    def __init__(self, config: AssistAIConfig | None = None) -> None:
        self.config: AssistAIConfig = config or AssistAIConfig.from_env()
        self._registry: ProviderRegistry = self._build_registry()
        # ``_http`` остаётся клиентом ProxyAPI; клиенты остальных провайдеров
        # создаются лениво в ``_client_for`` и закрываются в ``close``.
        self._http: httpx.AsyncClient = self._new_http_client(
            self._registry.get("proxyapi")
        )
        self._extra_clients: dict[str, httpx.AsyncClient] = {}
        self._tools: ToolRegistry = ToolRegistry()
        self._weather_client: OpenWeatherClient | None = None
        self._closed: bool = False
        self._register_builtin_tools()

    def _register_builtin_tools(self) -> None:
        self._weather_client = register_data_tools(
            self._tools,
            openweather_api_key=self.config.openweather_api_key,
            exchangerate_api_key=self.config.exchangerate_api_key,
        )
        self.register_tool(GenerateImageTool(self))
        # Late import: media_tools зависит от AssistAIClient.
        from app.media.media_tools import register_media_tools

        register_media_tools(self)

    def register_tool(self, tool: BaseTool) -> None:
        """Добавляет инструмент в реестр. Повторное имя заменяет предыдущий инструмент."""
        self._tools.register(tool)

    def get_tools_schema(self) -> list[dict[str, Any]]:
        """Возвращает схемы всех зарегистрированных инструментов для поля ``tools``."""
        return self._tools.schema()

    @property
    def _tools_registry(self) -> dict[str, BaseTool]:
        """Совместимость со старым доступом к словарю инструментов."""
        return self._tools._tools

    @staticmethod
    def _parse_tool_arguments(raw: Any) -> dict[str, Any]:
        """Совместимость: раньше метод жил на ``AssistAIClient``."""
        return ToolRegistry.parse_arguments(raw)
