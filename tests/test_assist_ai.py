"""Unit-тесты AssistAIClient. HTTP только через respx, без живых API."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from respx import MockRouter

from app.bot.runtime import _extract_generated_images
from app.llm.assist_ai import AssistAIClient, AssistAIError, ImageModelScope
from app.llm.chat_runtime import (
    _attach_chat_tools,
    _gpt6_chat_omit_tools,
    _gpt6_chat_tools_need_reasoning_none,
    _length_mark_suffix,
    _ToolCallBuilder,
)
from app.llm.tooling import BaseTool

CHAT_URL = "https://api.proxyapi.ru/v1/chat/completions"


class _RecordingTool(BaseTool):
    """Инструмент-заглушка: запоминает аргументы и не ходит в сеть."""

    def __init__(self, tool_name: str) -> None:
        self._name = tool_name
        self.calls: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "Тестовый инструмент."

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return json.dumps({"ok": True, "echo": kwargs}, ensure_ascii=False)


def _completion(message: dict[str, Any], finish_reason: str) -> dict[str, Any]:
    return {
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": message,
            }
        ]
    }


def _tool_call(call_id: str, name: str, arguments: str) -> dict[str, Any]:
    return _completion(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            ],
        },
        "tool_calls",
    )


def _sse(events: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for event in events:
        lines.append("data: " + json.dumps(event, ensure_ascii=False))
        lines.append("")
    lines.append("data: [DONE]")
    lines.append("")
    return "\n".join(lines)


def _stream_delta(
    delta: dict[str, Any], finish_reason: str | None = None
) -> dict[str, Any]:
    return {
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ]
    }


def test_tool_call_builder_keeps_parallel_calls_when_index_is_omitted() -> None:
    """Повтор имени не склеивается; продолжение без index идёт в последний слот."""
    builder = _ToolCallBuilder()
    builder.add(
        {
            "index": 0,
            "id": "call_a",
            "function": {"name": "get_weather", "arguments": "{"},
        }
    )
    builder.add(
        {
            "index": 1,
            "id": "call_b",
            "function": {"name": "get_time", "arguments": ""},
        }
    )
    builder.add({"function": {"name": "get_time", "arguments": '{"tz":"UTC"}'}})
    builder.add(
        {
            "index": 0,
            "function": {"name": "get_weather", "arguments": '"city":"Омск"}'},
        }
    )
    calls = builder.calls()
    assert [call["id"] for call in calls] == ["call_a", "call_b"]
    assert calls[0]["function"]["name"] == "get_weather"
    assert calls[0]["function"]["arguments"] == '{"city":"Омск"}'
    assert calls[1]["function"]["name"] == "get_time"
    assert calls[1]["function"]["arguments"] == '{"tz":"UTC"}'


def test_length_mark_is_one_trailing_line() -> None:
    assert _length_mark_suffix("Фраза обрывает") == "\n…ответ обрезан по лимиту длины"
    assert _length_mark_suffix("Фраза обрывает\n") == "…ответ обрезан по лимиту длины"
    assert _length_mark_suffix("уже\n…ответ обрезан по лимиту длины") == ""
    assert _length_mark_suffix("") == "…ответ обрезан по лимиту длины"


@pytest.mark.asyncio
async def test_stream_chat_marks_length_finish(
    assist_client: AssistAIClient,
    respx_mock: MockRouter,
) -> None:
    """Обрезка по max_tokens дописывает одну строку и в чат, и в историю хода."""
    respx_mock.post(CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            text=_sse([_stream_delta({"content": "Фраза обрывает"}, "length")]),
        )
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": "продолжай"}]
    deltas = [
        delta async for delta in assist_client.stream_chat(messages, use_tools=False)
    ]
    text = "".join(delta.text for delta in deltas if delta.kind == "content")
    assert text == "Фраза обрывает\n…ответ обрезан по лимиту длины"
    assert messages[-1] == {"role": "assistant", "content": text}


@pytest.mark.asyncio
async def test_stream_chat_stop_has_no_length_mark(
    assist_client: AssistAIClient,
    respx_mock: MockRouter,
) -> None:
    respx_mock.post(CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            text=_sse([_stream_delta({"content": "Целая фраза."}, "stop")]),
        )
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": "коротко"}]
    deltas = [
        delta async for delta in assist_client.stream_chat(messages, use_tools=False)
    ]
    text = "".join(delta.text for delta in deltas if delta.kind == "content")
    assert text == "Целая фраза."
    assert messages[-1]["content"] == "Целая фраза."


@pytest.mark.asyncio
async def test_stream_chat_does_not_mark_length_on_a_tool_round(
    assist_client: AssistAIClient,
    respx_mock: MockRouter,
) -> None:
    """Пометка только у финального ответа. Промежуточный tool-ход её не получает."""
    assist_client.register_tool(_RecordingTool("get_weather_by_city"))
    first = _sse(
        [
            _stream_delta({"content": "Сейчас узнаю. "}),
            _stream_delta(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_len_1",
                            "function": {
                                "name": "get_weather_by_city",
                                "arguments": '{"city_name": "Омск"}',
                            },
                        }
                    ]
                },
                "length",
            ),
        ]
    )
    second = _sse([_stream_delta({"content": "В Омске ясно."}, "stop")])
    respx_mock.post(CHAT_URL).mock(
        side_effect=[
            httpx.Response(200, text=first),
            httpx.Response(200, text=second),
        ]
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": "погода"}]
    deltas = [delta async for delta in assist_client.stream_chat(messages)]
    text = "".join(delta.text for delta in deltas if delta.kind == "content")
    assert text == "Сейчас узнаю. В Омске ясно."
    assert messages[-1]["content"] == "В Омске ясно."


def test_gpt6_tools_require_reasoning_none() -> None:
    """luna/sol/terra → reasoning_effort=none при tools; omit-список пуст."""
    assert _gpt6_chat_tools_need_reasoning_none("openai/gpt-6-luna")
    assert _gpt6_chat_tools_need_reasoning_none("gpt-6-sol")
    assert _gpt6_chat_tools_need_reasoning_none("openai/gpt-6.1-terra")
    assert not _gpt6_chat_omit_tools("openai/gpt-6-sol")
    assert not _gpt6_chat_omit_tools("openai/gpt-6.1-terra")
    assert not _gpt6_chat_omit_tools("openai/gpt-6-luna")
    assert not _gpt6_chat_tools_need_reasoning_none("openai/gpt-4o")

    tools = [{"type": "function", "function": {"name": "ping"}}]
    gpt6: dict[str, Any] = {"model": "openai/gpt-6-luna"}
    _attach_chat_tools(gpt6, tools, "openai/gpt-6-luna")
    assert gpt6["tools"] == tools
    assert gpt6["tool_choice"] == "auto"
    assert gpt6["reasoning_effort"] == "none"

    sol: dict[str, Any] = {"model": "openai/gpt-6-sol"}
    _attach_chat_tools(sol, tools, "openai/gpt-6-sol")
    assert sol["tools"] == tools
    assert sol["reasoning_effort"] == "none"

    terra: dict[str, Any] = {"model": "openai/gpt-6.1-terra"}
    _attach_chat_tools(terra, tools, "openai/gpt-6.1-terra")
    assert terra["tools"] == tools
    assert terra["reasoning_effort"] == "none"

    other: dict[str, Any] = {"model": "openai/gpt-4o"}
    _attach_chat_tools(other, tools, "openai/gpt-4o")
    assert other["tools"] == tools
    assert "reasoning_effort" not in other

    empty: dict[str, Any] = {"model": "openai/gpt-6-luna"}
    _attach_chat_tools(empty, [], "openai/gpt-6-luna")
    assert "tools" not in empty
    assert "reasoning_effort" not in empty


@pytest.mark.asyncio
async def test_stream_chat_gpt6_sol_sends_tools_with_none(
    assist_client: AssistAIClient,
    respx_mock: MockRouter,
) -> None:
    """gpt-6-sol на chat/completions шлёт tools с reasoning_effort=none."""
    route = respx_mock.post(CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            text=_sse([_stream_delta({"content": "ок"}, "stop")]),
        )
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": "привет"}]
    deltas = [
        delta
        async for delta in assist_client.stream_chat(messages, model="openai/gpt-6-sol")
    ]
    assert "".join(d.text for d in deltas if d.kind == "content") == "ок"
    body = json.loads(route.calls[0].request.content)
    assert body["model"] == "openai/gpt-6-sol"
    assert body.get("tools")
    assert body["reasoning_effort"] == "none"
    assert body["stream_options"] == {"include_usage": True}


@pytest.mark.asyncio
async def test_stream_chat_gpt6_luna_sends_reasoning_none(
    assist_client: AssistAIClient,
    respx_mock: MockRouter,
) -> None:
    """Живой payload для gpt-6-luna с tools содержит reasoning_effort=none."""
    route = respx_mock.post(CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            text=_sse([_stream_delta({"content": "4"}, "stop")]),
        )
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": "2+2?"}]
    deltas = [
        delta
        async for delta in assist_client.stream_chat(
            messages, model="openai/gpt-6-luna"
        )
    ]
    assert "".join(d.text for d in deltas if d.kind == "content") == "4"
    body = json.loads(route.calls[0].request.content)
    assert body["model"] == "openai/gpt-6-luna"
    assert body.get("tools")
    assert body["reasoning_effort"] == "none"
    assert body["stream_options"] == {"include_usage": True}


def test_tool_arguments_parsing_robustness() -> None:
    """Строка JSON и уже готовый dict принимаются, список аргументов отвергается."""
    parsed = AssistAIClient._parse_tool_arguments(
        '{"from_currency": "USD", "amount": 100}'
    )
    assert parsed == {"from_currency": "USD", "amount": 100}

    ready = {"to_currency": "RUB", "amount": 1}
    assert AssistAIClient._parse_tool_arguments(ready) is ready
    assert AssistAIClient._parse_tool_arguments(None) == {}
    assert AssistAIClient._parse_tool_arguments("") == {}

    with pytest.raises(ValueError, match="JSON-объектом"):
        AssistAIClient._parse_tool_arguments('["USD", "RUB"]')
    with pytest.raises(ValueError, match="Неподдерживаемый тип"):
        AssistAIClient._parse_tool_arguments(["USD", "RUB"])


@pytest.mark.asyncio
async def test_chat_completion_tool_call_execution(
    assist_client: AssistAIClient,
    respx_mock: MockRouter,
) -> None:
    """Вызов convert_currency дописывает role=tool и второй запрос возвращает текст."""
    arguments = {"from_currency": "USD", "to_currency": "RUB", "amount": 100}
    tool_body = json.dumps(
        {
            "from_currency": "USD",
            "to_currency": "RUB",
            "amount": 100,
            "rate": 90.0,
            "result": 9000.0,
        },
        ensure_ascii=False,
    )
    recorded = _RecordingTool("convert_currency")

    async def execute(**kwargs: Any) -> str:
        recorded.calls.append(kwargs)
        return tool_body

    assist_client._tools_registry["convert_currency"].execute = execute  # type: ignore[method-assign]

    route = respx_mock.post(CHAT_URL).mock(
        side_effect=[
            httpx.Response(
                200,
                json=_tool_call("call_fx_1", "convert_currency", json.dumps(arguments)),
            ),
            httpx.Response(
                200,
                json=_completion(
                    {"role": "assistant", "content": "100 долларов — это 9000 рублей."},
                    "stop",
                ),
            ),
        ]
    )
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Сколько 100 долларов в рублях?"}
    ]

    reply = await assist_client.chat_completion(messages)

    assert reply == "100 долларов — это 9000 рублей."
    assert route.call_count == 2
    assert recorded.calls == [arguments]
    tool_messages = [item for item in messages if item.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "call_fx_1"
    assert json.loads(tool_messages[0]["content"])["result"] == 9000.0
    second_messages = json.loads(route.calls[1].request.content)["messages"]
    assert any(item.get("role") == "tool" for item in second_messages)
    assert [item["role"] for item in messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]


@pytest.mark.asyncio
async def test_stream_chat_tool_call_stitching(
    assist_client: AssistAIClient,
    respx_mock: MockRouter,
) -> None:
    """Аргументы склеиваются, повтор имени не удваивается, нет index — прошлый слот."""
    fragments = [
        {
            "index": 0,
            "id": "call_w_1",
            "function": {
                "name": "get_weather_by_city",
                "arguments": '{"city_na',
            },
        },
        {
            "function": {
                "name": "get_weather_by_city",
                "arguments": 'me": "Москва"}',
            },
        },
    ]
    builder = _ToolCallBuilder()
    for fragment in fragments:
        builder.add(fragment)
    assert builder.calls()[0]["function"]["name"] == "get_weather_by_city"
    assert builder.calls()[0]["function"]["arguments"] == '{"city_name": "Москва"}'

    stub = _RecordingTool("get_weather_by_city")
    assist_client.register_tool(stub)
    first_stream = _sse(
        [
            _stream_delta({"tool_calls": [fragments[0]]}),
            _stream_delta({"tool_calls": [fragments[1]]}),
            _stream_delta({}, "tool_calls"),
        ]
    )
    second_stream = _sse(
        [
            _stream_delta({"content": "В Москве "}),
            _stream_delta({"content": "ясно."}, "stop"),
        ]
    )
    route = respx_mock.post(CHAT_URL).mock(
        side_effect=[
            httpx.Response(200, text=first_stream),
            httpx.Response(200, text=second_stream),
        ]
    )
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Какая погода в Москве?"}
    ]

    deltas = [delta async for delta in assist_client.stream_chat(messages)]

    assert route.call_count == 2
    for call in route.calls:
        body = json.loads(call.request.content)
        assert body["stream_options"] == {"include_usage": True}
    assert stub.calls == [{"city_name": "Москва"}]
    tool_deltas = [delta for delta in deltas if delta.kind == "tool"]
    assert tool_deltas
    assert tool_deltas[0].text.startswith("get_weather_by_city")
    assert (
        "".join(delta.text for delta in deltas if delta.kind == "content")
        == "В Москве ясно."
    )
    tool_message = next(item for item in messages if item.get("tool_calls"))
    assert tool_message["tool_calls"][0]["function"]["name"] == "get_weather_by_city"


@pytest.mark.asyncio
async def test_stream_chat_emits_tool_delta_before_execute(
    assist_client: AssistAIClient,
    respx_mock: MockRouter,
) -> None:
    """Статус чата должен успеть обновиться до долгого generate_image / видео."""

    class _MarkTool(_RecordingTool):
        def __init__(self) -> None:
            super().__init__("get_weather_by_city")
            self.ran = False

        async def execute(self, **kwargs: Any) -> str:
            self.ran = True
            return await super().execute(**kwargs)

    tool = _MarkTool()
    assist_client.register_tool(tool)
    stream = _sse(
        [
            _stream_delta(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_w_1",
                            "function": {
                                "name": "get_weather_by_city",
                                "arguments": '{"city_name": "Москва"}',
                            },
                        }
                    ]
                },
                "tool_calls",
            )
        ]
    )
    followup = _sse([_stream_delta({"content": "ясно"}, "stop")])
    respx_mock.post(CHAT_URL).mock(
        side_effect=[
            httpx.Response(200, text=stream),
            httpx.Response(200, text=followup),
        ]
    )
    seen_before_run = False
    async for delta in assist_client.stream_chat(
        [{"role": "user", "content": "погода"}]
    ):
        if delta.kind == "tool":
            seen_before_run = not tool.ran
            assert delta.text.startswith("get_weather_by_city")
    assert seen_before_run is True
    assert tool.ran is True
    assert tool.calls == [{"city_name": "Москва"}]


@pytest.mark.asyncio
async def test_stream_chat_keeps_reasoning_only_for_tool_rounds(
    assist_client: AssistAIClient,
    respx_mock: MockRouter,
) -> None:
    """reasoning_content не yield’ится и не остаётся в финальном assistant."""
    stub = _RecordingTool("get_weather_by_city")
    assist_client.register_tool(stub)
    first_stream = _sse(
        [
            _stream_delta({"reasoning_content": "сначала подумаю"}),
            _stream_delta(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_r_1",
                            "function": {
                                "name": "get_weather_by_city",
                                "arguments": '{"city_name": "Казань"}',
                            },
                        }
                    ]
                }
            ),
            _stream_delta({}, "tool_calls"),
        ]
    )
    second_stream = _sse(
        [
            _stream_delta({"reasoning_content": "уже не в чат"}),
            _stream_delta({"content": "В Казани "}),
            _stream_delta({"content": "облачно."}, "stop"),
        ]
    )
    respx_mock.post(CHAT_URL).mock(
        side_effect=[
            httpx.Response(200, text=first_stream),
            httpx.Response(200, text=second_stream),
        ]
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": "Погода в Казани?"}]

    deltas = [delta async for delta in assist_client.stream_chat(messages)]

    assert all(delta.kind in {"content", "tool"} for delta in deltas)
    assert "".join(delta.text for delta in deltas if delta.kind == "content") == (
        "В Казани облачно."
    )
    assert not any(
        "подумаю" in delta.text or "не в чат" in delta.text for delta in deltas
    )

    tool_assistant = next(item for item in messages if item.get("tool_calls"))
    assert tool_assistant["reasoning_content"] == "сначала подумаю"

    final_assistant = messages[-1]
    assert final_assistant["role"] == "assistant"
    assert final_assistant["content"] == "В Казани облачно."
    assert "reasoning_content" not in final_assistant
    assert "reasoning" not in final_assistant


@pytest.mark.asyncio
async def test_http_timeout_handling(
    assist_client: AssistAIClient,
    respx_mock: MockRouter,
) -> None:
    """Таймаут httpx не выходит наружу: _post_json поднимает AssistAIError."""
    respx_mock.post(CHAT_URL).mock(side_effect=httpx.ReadTimeout("timed out"))

    with pytest.raises(AssistAIError) as captured:
        await assist_client._post_json(
            "chat/completions", {"model": "openai/gpt-4o", "messages": []}
        )

    error = captured.value
    assert isinstance(error.__cause__, httpx.TimeoutException)
    assert "Превышено время ожидания" in error.message
    assert "chat/completions" in error.message


@pytest.mark.asyncio
async def test_max_tool_iterations_limit(
    assist_client: AssistAIClient,
    respx_mock: MockRouter,
) -> None:
    """Шестой ответ с tool_calls не запрашивается: на пятом круге цикл прерывается."""
    assert assist_client.config.max_tool_iterations == 5
    stub = _RecordingTool("convert_currency")
    assist_client.register_tool(stub)
    responses = [
        httpx.Response(
            200,
            json=_tool_call(
                f"call_{index}",
                "convert_currency",
                '{"from_currency": "USD", "to_currency": "RUB", "amount": 1}',
            ),
        )
        for index in range(6)
    ]
    route = respx_mock.post(CHAT_URL).mock(side_effect=responses)
    messages: list[dict[str, Any]] = [{"role": "user", "content": "Курс доллара"}]

    with pytest.raises(AssistAIError, match="Превышен лимит вызовов инструментов: 5"):
        await assist_client.chat_completion(messages)

    assert route.call_count == 5
    assert len(stub.calls) == 5
    assert sum(1 for item in messages if item.get("role") == "tool") == 5


@pytest.mark.asyncio
async def test_generate_image_tool_packs_url_and_base64(
    assist_client: AssistAIClient,
    respx_mock: MockRouter,
) -> None:
    """Инструмент не бросает исключения и помечает url отдельно от base64."""
    respx_mock.post("https://api.proxyapi.ru/v1/images/generations").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {"url": "https://cdn.example/cat.png"},
                    {"b64_json": "aGVsbG8="},
                ]
            },
        )
    )
    tool = assist_client._tools_registry["generate_image"]

    with ImageModelScope(None) as scope:
        payload = json.loads(await tool.execute(prompt="кот в шляпе", size="1024x1024"))

    assert payload["status"] == "success"
    assert payload["prompt"] == "кот в шляпе"
    assert payload["image_count"] == 2
    assert payload["images"] == [{"type": "url"}, {"type": "base64"}]
    assert payload["reference_indexes"] == [0, 1]
    assert "aGVsbG8=" not in json.dumps(payload)
    assert scope.images == [
        {
            "type": "url",
            "value": "https://cdn.example/cat.png",
            "model": "openai/gpt-image-2.6-flare",
        },
        {
            "type": "base64",
            "value": "aGVsbG8=",
            "model": "openai/gpt-image-2.6-flare",
        },
    ]


@pytest.mark.asyncio
async def test_generate_image_tool_returns_error_json(
    assist_client: AssistAIClient,
    respx_mock: MockRouter,
) -> None:
    """Сбой ProxyAPI остаётся JSON-ошибкой для модели, а не исключением."""
    respx_mock.post("https://api.proxyapi.ru/v1/images/generations").mock(
        return_value=httpx.Response(500, text="boom")
    )
    tool = assist_client._tools_registry["generate_image"]

    payload = json.loads(await tool.execute(prompt="схема"))

    assert "error" in payload
    assert "images" not in payload


@pytest.mark.asyncio
async def test_image_model_scope_used_when_tool_omits_model(
    assist_client: AssistAIClient,
) -> None:
    """Модель персоны подставляется, если инструмент вызван без model."""
    seen: dict[str, str | None] = {}

    async def _fake_generate(
        prompt: str,
        model: str | None = None,
        size: str = "1024x1024",
        background: str | None = None,
    ) -> list[str]:
        seen["model"] = model
        seen["prompt"] = prompt
        seen["background"] = background
        return ["https://cdn.example/cat.png"]

    assist_client.generate_image = _fake_generate  # type: ignore[method-assign]
    tool = assist_client._tools_registry["generate_image"]
    with ImageModelScope("openai/gpt-image-2.6-flare"):
        payload = json.loads(await tool.execute(prompt="кот"))

    assert seen["model"] == "openai/gpt-image-2.6-flare"
    assert seen["background"] is None
    assert payload["status"] == "success"

    with ImageModelScope("openai/gpt-image-2.6-flare", background="transparent"):
        payload = json.loads(await tool.execute(prompt="кот"))
    assert seen["background"] == "transparent"
    assert payload["status"] == "success"


@pytest.mark.asyncio
async def test_image_size_scope_used_when_tool_omits_size(
    assist_client: AssistAIClient,
) -> None:
    """Размер персоны доходит до генерации и не затирается аргументом инструмента."""
    seen: dict[str, str | None] = {}
    sizes: list[str] = []

    async def _fake_generate(
        prompt: str,
        model: str | None = None,
        size: str = "1024x1024",
        background: str | None = None,
    ) -> list[str]:
        sizes.append(size)
        seen["model"] = model
        seen["background"] = background
        return ["https://cdn.example/cat.png"]

    assist_client.generate_image = _fake_generate  # type: ignore[method-assign]
    tool = assist_client._tools_registry["generate_image"]
    with ImageModelScope("black-forest-labs/flux-3-image", "2K"):
        explicit = json.loads(await tool.execute(prompt="кот", size="1K"))
        scoped = json.loads(await tool.execute(prompt="кот"))

    assert explicit["status"] == "success"
    assert scoped["status"] == "success"
    assert sizes == ["2K", "2K"]
    assert seen["model"] == "black-forest-labs/flux-3-image"


@pytest.mark.asyncio
async def test_persona_openai_size_reaches_generate_despite_tool_default(
    assist_client: AssistAIClient,
) -> None:
    """Кнопка 1024x1792 не подменяется дефолтом схемы 1024x1024."""
    sizes: list[str] = []

    async def _fake_generate(
        prompt: str,
        model: str | None = None,
        size: str = "1024x1024",
        background: str | None = None,
    ) -> list[str]:
        sizes.append(size)
        return ["https://cdn.example/cat.png"]

    assist_client.generate_image = _fake_generate  # type: ignore[method-assign]
    tool = assist_client._tools_registry["generate_image"]
    with ImageModelScope("openai/gpt-image-2.6-flare", "1024x1792"):
        payload = json.loads(await tool.execute(prompt="маяк", size="1024x1024"))

    assert payload["status"] == "success"
    assert sizes == ["1024x1792"]


@pytest.mark.asyncio
async def test_flux_pixel_size_becomes_resolution_tier(
    assist_client: AssistAIClient,
    respx_mock: MockRouter,
) -> None:
    """FLUX не принимает 1024x1024: в запрос уходит ярус 1K."""
    route = respx_mock.post("https://api.proxyapi.ru/v1/images/generations").mock(
        return_value=httpx.Response(
            200, json={"data": [{"url": "https://cdn.example/cat.png"}]}
        )
    )

    images = await assist_client.generate_image(
        "кот",
        model="black-forest-labs/flux-3-image",
        size="1024x1024",
    )

    body = json.loads(route.calls.last.request.content)
    assert body["size"] == "1K"
    assert "aspect_ratio" not in body
    assert images == ["https://cdn.example/cat.png"]


@pytest.mark.asyncio
async def test_openai_image_keeps_pixel_size(
    assist_client: AssistAIClient,
    respx_mock: MockRouter,
) -> None:
    """Пиксельный size остаётся у моделей, которым он нужен."""
    route = respx_mock.post("https://api.proxyapi.ru/v1/images/generations").mock(
        return_value=httpx.Response(
            200, json={"data": [{"url": "https://cdn.example/cat.png"}]}
        )
    )

    await assist_client.generate_image(
        "кот",
        model="openai/gpt-image-2.6-flare",
        size="1024x1024",
    )

    body = json.loads(route.calls.last.request.content)
    assert body["size"] == "1024x1024"
    assert "aspect_ratio" not in body


@pytest.mark.asyncio
async def test_qwen_image_sends_aspect_ratio(
    assist_client: AssistAIClient,
    respx_mock: MockRouter,
) -> None:
    """Qwen получает aspect_ratio, а не только OpenAI size."""
    route = respx_mock.post("https://api.proxyapi.ru/v1/images/generations").mock(
        return_value=httpx.Response(
            200, json={"data": [{"url": "https://cdn.example/cat.png"}]}
        )
    )

    await assist_client.generate_image(
        "кот",
        model="qwen/qwen-image-2.1-turbo",
        size="16:9",
    )

    body = json.loads(route.calls.last.request.content)
    assert body["aspect_ratio"] == "16:9"
    assert body.get("size") is None


@pytest.mark.asyncio
async def test_seedream_uses_size_preset_or_width_height(
    assist_client: AssistAIClient,
    respx_mock: MockRouter,
) -> None:
    """Seedream: ярус → size_preset, пиксели → width/height."""
    route = respx_mock.post("https://api.proxyapi.ru/v1/images/generations").mock(
        return_value=httpx.Response(
            200, json={"data": [{"url": "https://cdn.example/cat.png"}]}
        )
    )

    await assist_client.generate_image(
        "кот",
        model="bytedance-seed/seedream-5-0-flash",
        size="2K",
    )
    preset_body = json.loads(route.calls.last.request.content)
    assert preset_body["size_preset"] == "2K"
    assert "size" not in preset_body

    await assist_client.generate_image(
        "кот",
        model="bytedance-seed/seedream-5-0-flash",
        size="1280x720",
    )
    pixel_body = json.loads(route.calls.last.request.content)
    assert pixel_body["width"] == 1280
    assert pixel_body["height"] == 720
    assert "size" not in pixel_body


@pytest.mark.asyncio
async def test_nano_banana_size_is_image_resolution(
    assist_client: AssistAIClient,
    respx_mock: MockRouter,
) -> None:
    """Nano Banana передаёт image_resolution (1K/2K/4K)."""
    route = respx_mock.post("https://api.proxyapi.ru/v1/images/generations").mock(
        return_value=httpx.Response(
            200, json={"data": [{"url": "https://cdn.example/cat.png"}]}
        )
    )

    await assist_client.generate_image(
        "логотип",
        model="google/gemini-nano-banana-2.1",
        size="1K",
    )

    body = json.loads(route.calls.last.request.content)
    assert body["image_resolution"] == "1K"
    assert "quality" not in body


def test_extract_generated_images_matches_tool_call_id() -> None:
    """Бот берёт только картинки вызова generate_image, не чужие tool-сообщения."""
    messages: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_img",
                    "type": "function",
                    "function": {"name": "generate_image", "arguments": "{}"},
                },
                {
                    "id": "call_fx",
                    "type": "function",
                    "function": {"name": "convert_currency", "arguments": "{}"},
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_fx",
            "content": json.dumps(
                {"images": [{"type": "url", "value": "https://nope"}]}
            ),
        },
        {
            "role": "tool",
            "tool_call_id": "call_img",
            "content": json.dumps(
                {
                    "status": "success",
                    "images": [{"type": "url", "value": "https://cdn.example/cat.png"}],
                }
            ),
        },
        {"role": "tool", "tool_call_id": "call_img", "content": "не json"},
    ]

    assert _extract_generated_images(messages) == [
        {"type": "url", "value": "https://cdn.example/cat.png"}
    ]


def test_paid_demo_lives_outside_the_package() -> None:
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    package = ast.parse(
        (root / "app" / "llm" / "assist_ai.py").read_text(encoding="utf-8")
    )
    package_names = {
        node.name
        for node in package.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    assert "main" not in package_names
    demo = ast.parse(
        (root / "scripts" / "demo_assist_ai.py").read_text(encoding="utf-8")
    )
    demo_names = {
        node.name for node in demo.body if isinstance(node, ast.AsyncFunctionDef)
    }
    assert "main" in demo_names
