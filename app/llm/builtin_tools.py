"""Встроенные data-tools: погода OpenWeather и конвертация валют."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Any

import structlog

from app.llm.exchangerate import Conversion, ExchangeAPIError, get_exchange_rate
from app.llm.openweather import (
    AirPollutionData,
    CurrentWeather,
    DailyForecast,
    OpenWeatherAPIError,
    OpenWeatherClient,
)
from app.llm.tooling import BaseTool, ToolRegistry, _error_json, _exception_text

logger = structlog.get_logger("AssistAI.builtin_tools")


def _parse_city(raw: Any, key: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"Параметр {key} обязателен и должен быть непустой строкой.")
    return raw.strip()


def _parse_country(raw: Any) -> str | None:
    if raw is None or raw == "":
        return None
    if not isinstance(raw, str):
        raise ValueError("country_code должен быть строкой, например RU.")
    stripped = raw.strip()
    return stripped or None


def _parse_bool(raw: Any, *, default: bool) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        lowered = raw.strip().lower()
        if lowered in {"true", "1", "yes", "да"}:
            return True
        if lowered in {"false", "0", "no", "нет"}:
            return False
    raise ValueError("with_forecast должен быть true или false.")


def _parse_coord(raw: Any, name: str, low: float, high: float) -> float:
    if raw is None or isinstance(raw, bool):
        raise ValueError(f"Параметр {name} обязателен и должен быть числом.")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Параметр {name} должен быть числом.") from exc
    if not low <= value <= high:
        raise ValueError(f"Параметр {name} должен быть от {low} до {high}.")
    return value


def _parse_days(raw: Any) -> int:
    if raw is None:
        return 5
    if isinstance(raw, bool):
        raise ValueError("days должен быть целым числом от 1 до 5.")
    if isinstance(raw, float) and not raw.is_integer():
        raise ValueError("days должен быть целым числом от 1 до 5.")
    try:
        days = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("days должен быть целым числом от 1 до 5.") from exc
    if days < 1 or days > 5:
        raise ValueError("days должен быть от 1 до 5.")
    return days


def _local_stamp(timestamp: int, timezone_offset: int, with_time: bool) -> str:
    moment = datetime.fromtimestamp(
        timestamp, timezone(timedelta(seconds=timezone_offset))
    )
    if with_time:
        return moment.strftime("%Y-%m-%d %H:%M")
    return moment.date().isoformat()


def _weather_payload(weather: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """Текущая погода и, если клиент её приложил, суточный прогноз."""
    current_raw = weather.get("current")
    current = CurrentWeather.from_owm(
        current_raw if isinstance(current_raw, dict) else {}
    )
    payload: dict[str, Any] = {
        "latitude": weather.get("lat"),
        "longitude": weather.get("lon"),
        "temperature": current.temp,
        "feels_like": current.feels_like,
        "description": current.description,
        "humidity": current.humidity,
        "wind_speed": current.wind_speed,
    }
    daily = weather.get("daily")
    if isinstance(daily, list):
        offset = int(weather.get("timezone") or 0)
        payload["daily"] = [
            _daily_payload(item, offset) for item in daily if isinstance(item, dict)
        ]
    payload.update(extra)
    return payload


def _daily_payload(item: dict[str, Any], timezone_offset: int) -> dict[str, Any]:
    day = DailyForecast.from_owm(item)
    return {
        "date": _local_stamp(day.dt, timezone_offset, with_time=False),
        "temp_day": day.temp_day,
        "temp_night": day.temp_night,
        "description": day.description,
    }


def _forecast_payload(forecast: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """Сжимает 3-часовой прогноз до полей, которые нужны ответу модели."""
    offset = int(forecast.get("timezone") or 0)
    hours: list[dict[str, Any]] = []
    for item in forecast.get("list") or []:
        if not isinstance(item, dict):
            continue
        main_raw = item.get("main")
        main = main_raw if isinstance(main_raw, dict) else {}
        wind_raw = item.get("wind")
        wind = wind_raw if isinstance(wind_raw, dict) else {}
        weather_raw = item.get("weather")
        weather_list = weather_raw if isinstance(weather_raw, list) else []
        first_weather = weather_list[0] if weather_list else None
        weather_info = first_weather if isinstance(first_weather, dict) else {}
        hours.append(
            {
                "time_local": _local_stamp(
                    int(item.get("dt") or 0), offset, with_time=True
                ),
                "temperature": main.get("temp"),
                "feels_like": main.get("feels_like"),
                "humidity": main.get("humidity"),
                "wind_speed": wind.get("speed"),
                "description": weather_info.get("description"),
                "precipitation_probability": item.get("pop"),
            }
        )
    return {
        "latitude": forecast.get("lat"),
        "longitude": forecast.get("lon"),
        "hours": hours,
        **extra,
    }


def _air_payload(data: dict[str, Any]) -> dict[str, Any]:
    items = data.get("list")
    first = (
        items[0]
        if isinstance(items, list) and items and isinstance(items[0], dict)
        else {}
    )
    parsed = AirPollutionData.from_owm(first)
    components = (
        first.get("components") if isinstance(first.get("components"), dict) else {}
    )
    return {
        "aqi": parsed.aqi,
        "aqi_scale": "1 — хорошо, 5 — опасно",
        "pm2_5": parsed.pm2_5,
        "pm10": parsed.pm10,
        "components": components,
    }


_CITY_NAME_SCHEMA: dict[str, Any] = {
    "type": "string",
    "description": "Название города, например Москва или Irkutsk.",
}
_COUNTRY_CODE_SCHEMA: dict[str, Any] = {
    "type": "string",
    "description": (
        "Необязательный код страны ISO 3166-1 alpha-2, например RU. "
        "Передавай, если город встречается в нескольких странах."
    ),
}
_LAT_SCHEMA: dict[str, Any] = {
    "type": "number",
    "minimum": -90,
    "maximum": 90,
    "description": "Широта в градусах.",
}
_LON_SCHEMA: dict[str, Any] = {
    "type": "number",
    "minimum": -180,
    "maximum": 180,
    "description": "Долгота в градусах.",
}
_FORECAST_FLAG_SCHEMA: dict[str, Any] = {
    "type": "boolean",
    "description": (
        "true — текущая погода и суточный прогноз до 5 дней. "
        "false — только текущая погода. По умолчанию true."
    ),
}


def _object_schema(
    properties: dict[str, Any],
    required: list[str],
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


class _OpenWeatherTool(BaseTool):
    """Общий контур ошибок для инструментов OpenWeather."""

    def __init__(self, client: OpenWeatherClient) -> None:
        self._client = client

    async def _guard(self, action: Callable[[], Awaitable[dict[str, Any]]]) -> str:
        try:
            payload = await action()
        except OpenWeatherAPIError as exc:
            logger.error("%s: %s", self.name, exc.message)
            return _error_json(exc.message)
        except Exception as exc:
            logger.exception("%s: непредвиденная ошибка", self.name)
            return _error_json(_exception_text(exc))
        logger.info("%s: ответ готов", self.name)
        return json.dumps(payload, ensure_ascii=False)


class WeatherByCityTool(_OpenWeatherTool):
    """Текущая погода и суточный прогноз по названию города."""

    @property
    def name(self) -> str:
        return "get_weather_by_city"

    @property
    def description(self) -> str:
        return (
            "Погода по названию города: температура в °C, ощущение, описание, "
            "влажность в процентах, ветер в м/с, координаты. "
            "При with_forecast=true добавляется суточный прогноз до 5 дней "
            "(дневная и ночная температура). "
            "Для прогноза на конкретный час используй get_forecast_by_city. "
            "Вызывай только если пользователь прямо спрашивает погоду в городе. "
            "Цифры бери из ответа функции, не из памяти."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return _object_schema(
            {
                "city_name": _CITY_NAME_SCHEMA,
                "country_code": _COUNTRY_CODE_SCHEMA,
                "with_forecast": _FORECAST_FLAG_SCHEMA,
            },
            ["city_name"],
        )

    async def execute(self, **kwargs: Any) -> str:
        """Вызывает ``OpenWeatherClient.get_weather_by_city``."""
        try:
            city_name = _parse_city(kwargs.get("city_name"), "city_name")
            country_code = _parse_country(kwargs.get("country_code"))
            with_forecast = _parse_bool(kwargs.get("with_forecast"), default=True)
        except ValueError as exc:
            return _error_json(str(exc))

        async def _load() -> dict[str, Any]:
            weather = await self._client.get_weather_by_city(
                city_name,
                country_code,
                with_forecast=with_forecast,
            )
            return _weather_payload(
                weather,
                city_name=city_name,
                country_code=country_code,
                with_forecast=with_forecast,
            )

        return await self._guard(_load)


class WeatherByCoordsTool(_OpenWeatherTool):
    """Текущая погода и суточный прогноз по координатам."""

    @property
    def name(self) -> str:
        return "get_weather_by_coords"

    @property
    def description(self) -> str:
        return (
            "Погода по широте и долготе. Возвращает те же поля, что "
            "get_weather_by_city. Параметр days задаёт длину суточного прогноза "
            "от 1 до 5 и работает только при with_forecast=true. "
            "Вызывай только если пользователь прямо спрашивает погоду по координатам."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return _object_schema(
            {
                "lat": _LAT_SCHEMA,
                "lon": _LON_SCHEMA,
                "days": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "description": "Число суток прогноза, от 1 до 5. По умолчанию 5.",
                },
                "with_forecast": _FORECAST_FLAG_SCHEMA,
            },
            ["lat", "lon"],
        )

    async def execute(self, **kwargs: Any) -> str:
        """Вызывает ``OpenWeatherClient.get_weather_by_coords``."""
        try:
            lat = _parse_coord(kwargs.get("lat"), "lat", -90, 90)
            lon = _parse_coord(kwargs.get("lon"), "lon", -180, 180)
            days = _parse_days(kwargs.get("days"))
            with_forecast = _parse_bool(kwargs.get("with_forecast"), default=True)
        except ValueError as exc:
            return _error_json(str(exc))

        async def _load() -> dict[str, Any]:
            weather = await self._client.get_weather_by_coords(
                lat,
                lon,
                days,
                with_forecast=with_forecast,
            )
            return _weather_payload(
                weather,
                latitude=lat,
                longitude=lon,
                days=days,
                with_forecast=with_forecast,
            )

        return await self._guard(_load)


class ForecastByCityTool(_OpenWeatherTool):
    """Прогноз с шагом 3 часа по названию города."""

    @property
    def name(self) -> str:
        return "get_forecast_by_city"

    @property
    def description(self) -> str:
        return (
            "Прогноз по городу на 5 дней с шагом 3 часа: местное время, "
            "температура, влажность, ветер и вероятность осадков. "
            "Вызывай, только если пользователь просит прогноз по часам или на часть дня. "
            "Для сводки по суткам достаточно get_weather_by_city."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return _object_schema(
            {
                "city_name": _CITY_NAME_SCHEMA,
                "country_code": _COUNTRY_CODE_SCHEMA,
            },
            ["city_name"],
        )

    async def execute(self, **kwargs: Any) -> str:
        """Вызывает ``OpenWeatherClient.get_forecast_by_city``."""
        try:
            city_name = _parse_city(kwargs.get("city_name"), "city_name")
            country_code = _parse_country(kwargs.get("country_code"))
        except ValueError as exc:
            return _error_json(str(exc))

        async def _load() -> dict[str, Any]:
            forecast = await self._client.get_forecast_by_city(city_name, country_code)
            return _forecast_payload(
                forecast,
                city_name=city_name,
                country_code=country_code,
            )

        return await self._guard(_load)


class ForecastByCoordsTool(_OpenWeatherTool):
    """Прогноз с шагом 3 часа по координатам."""

    @property
    def name(self) -> str:
        return "get_forecast_by_coords"

    @property
    def description(self) -> str:
        return (
            "Прогноз по координатам на 5 дней с шагом 3 часа. "
            "Поля те же, что у get_forecast_by_city. "
            "Вызывай только если пользователь прямо просит такой прогноз."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return _object_schema(
            {"lat": _LAT_SCHEMA, "lon": _LON_SCHEMA},
            ["lat", "lon"],
        )

    async def execute(self, **kwargs: Any) -> str:
        """Вызывает ``OpenWeatherClient.get_forecast_by_coords``."""
        try:
            lat = _parse_coord(kwargs.get("lat"), "lat", -90, 90)
            lon = _parse_coord(kwargs.get("lon"), "lon", -180, 180)
        except ValueError as exc:
            return _error_json(str(exc))

        async def _load() -> dict[str, Any]:
            forecast = await self._client.get_forecast_by_coords(lat, lon)
            return _forecast_payload(forecast, latitude=lat, longitude=lon)

        return await self._guard(_load)


class AirPollutionTool(_OpenWeatherTool):
    """Качество воздуха по координатам."""

    @property
    def name(self) -> str:
        return "get_air_pollution"

    @property
    def description(self) -> str:
        return (
            "Качество воздуха по координатам: индекс AQI от 1 (хорошо) до 5 (опасно), "
            "PM2.5, PM10 и остальные компоненты. "
            "Координаты можно взять из get_weather_by_city или get_weather_by_coords. "
            "Вызывай только если пользователь прямо спрашивает качество воздуха."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return _object_schema(
            {"lat": _LAT_SCHEMA, "lon": _LON_SCHEMA},
            ["lat", "lon"],
        )

    async def execute(self, **kwargs: Any) -> str:
        """Вызывает ``OpenWeatherClient.get_air_pollution``."""
        try:
            lat = _parse_coord(kwargs.get("lat"), "lat", -90, 90)
            lon = _parse_coord(kwargs.get("lon"), "lon", -180, 180)
        except ValueError as exc:
            return _error_json(str(exc))

        async def _load() -> dict[str, Any]:
            data = await self._client.get_air_pollution(lat, lon)
            payload = _air_payload(data)
            payload["latitude"] = lat
            payload["longitude"] = lon
            return payload

        return await self._guard(_load)


class CompareCitiesTool(_OpenWeatherTool):
    """Сравнение текущей погоды в двух городах."""

    @property
    def name(self) -> str:
        return "compare_cities"

    @property
    def description(self) -> str:
        return (
            "Сравнить текущую погоду в двух городах: температура, ощущение, "
            "описание и скорость ветра. Города запрашиваются параллельно. "
            "Вызывай только если пользователь прямо просит сравнить погоду."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return _object_schema(
            {
                "city1": {
                    "type": "string",
                    "description": "Первый город.",
                },
                "city2": {
                    "type": "string",
                    "description": "Второй город.",
                },
            },
            ["city1", "city2"],
        )

    async def execute(self, **kwargs: Any) -> str:
        """Вызывает ``OpenWeatherClient.compare_cities``."""
        try:
            city1 = _parse_city(kwargs.get("city1"), "city1")
            city2 = _parse_city(kwargs.get("city2"), "city2")
        except ValueError as exc:
            return _error_json(str(exc))

        async def _load() -> dict[str, Any]:
            comparison = await self._client.compare_cities(city1, city2)
            return (
                comparison if isinstance(comparison, dict) else {"result": comparison}
            )

        return await self._guard(_load)


class WeatherNotificationTool(_OpenWeatherTool):
    """Готовый текст погодного уведомления."""

    @property
    def name(self) -> str:
        return "format_notification"

    @property
    def description(self) -> str:
        return (
            "Собрать готовый текст уведомления о погоде в городе: температура, "
            "ветер, влажность, описание и качество воздуха. "
            "Вызывай только если пользователь прямо просит готовый текст уведомления о погоде."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return _object_schema(
            {
                "city_name": _CITY_NAME_SCHEMA,
                "country_code": _COUNTRY_CODE_SCHEMA,
            },
            ["city_name"],
        )

    async def execute(self, **kwargs: Any) -> str:
        """Вызывает ``OpenWeatherClient.format_notification`` и кладёт текст в JSON."""
        try:
            city_name = _parse_city(kwargs.get("city_name"), "city_name")
            country_code = _parse_country(kwargs.get("country_code"))
        except ValueError as exc:
            return _error_json(str(exc))

        async def _load() -> dict[str, Any]:
            text = await self._client.format_notification(city_name, country_code)
            return {
                "city_name": city_name,
                "country_code": country_code,
                "notification": text,
            }

        return await self._guard(_load)


_WEATHER_TOOLS: tuple[type[_OpenWeatherTool], ...] = (
    WeatherByCityTool,
    WeatherByCoordsTool,
    ForecastByCityTool,
    ForecastByCoordsTool,
    AirPollutionTool,
    CompareCitiesTool,
    WeatherNotificationTool,
)


class ExchangeRateTool(BaseTool):
    """Конвертация суммы между двумя валютами через ``get_exchange_rate``."""

    def __init__(self, api_key: str) -> None:
        key = api_key.strip()
        if not key:
            raise ValueError("Для ExchangeRateTool нужен непустой API-ключ.")
        self._api_key = key

    @property
    def name(self) -> str:
        return "convert_currency"

    @property
    def description(self) -> str:
        return (
            "Конвертировать сумму из одной валюты в другую по актуальному курсу. "
            "Коды валют — ISO 4217 из трёх букв, например USD, EUR, RUB. "
            "Возвращает курс за 1 единицу и итоговую сумму. "
            "Вызывай только если пользователь прямо просит курс или конвертацию, "
            "и не считай курс сам."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "from_currency": {
                    "type": "string",
                    "description": "Исходная валюта, код ISO 4217. Пример: USD.",
                },
                "to_currency": {
                    "type": "string",
                    "description": "Целевая валюта, код ISO 4217. Пример: RUB.",
                },
                "amount": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "description": "Положительная сумма в исходной валюте.",
                },
            },
            "required": ["from_currency", "to_currency", "amount"],
            "additionalProperties": False,
        }

    async def execute(self, **kwargs: Any) -> str:
        """Конвертирует сумму и сериализует ``Conversion`` в JSON.

        ``ExchangeAPIError`` возвращается модели текстом ``user_message``.
        Остальные исключения тоже становятся ``{"error": "..."}``.
        """
        try:
            conversion: Conversion = await get_exchange_rate(
                from_currency=str(kwargs["from_currency"]),
                to_currency=str(kwargs["to_currency"]),
                amount=float(kwargs["amount"]),
                api_key=self._api_key,
            )
        except ExchangeAPIError as exc:
            logger.error("convert_currency: %s", exc.user_message)
            return _error_json(exc.user_message)
        except KeyError as exc:
            missing = exc.args[0] if exc.args else "параметр"
            logger.error("convert_currency: не передан параметр %s", missing)
            return _error_json(f"Не передан обязательный параметр: {missing}")
        except (TypeError, ValueError) as exc:
            logger.error("convert_currency: %s", exc)
            return _error_json(_exception_text(exc))
        except Exception as exc:
            logger.exception("convert_currency: непредвиденная ошибка")
            return _error_json(_exception_text(exc))

        logger.info(
            "convert_currency: %s %s -> %s",
            conversion.amount,
            conversion.from_currency,
            conversion.to_currency,
        )
        return json.dumps(asdict(conversion), ensure_ascii=False)


def register_data_tools(
    registry: ToolRegistry,
    *,
    openweather_api_key: str = "",
    exchangerate_api_key: str = "",
) -> OpenWeatherClient | None:
    """Регистрирует weather/currency tools. Возвращает weather-клиент или None."""
    weather_client: OpenWeatherClient | None = None
    key = openweather_api_key.strip()
    if key:
        from app.llm.openweather import OWMConfig

        weather_client = OpenWeatherClient(OWMConfig(api_key=key))
        for tool_type in _WEATHER_TOOLS:
            registry.register(tool_type(weather_client))
    else:
        logger.info("OPENWEATHER_API_KEY не задан, инструменты погоды отключены")

    fx_key = exchangerate_api_key.strip()
    if fx_key:
        registry.register(ExchangeRateTool(fx_key))
    else:
        logger.info("EXCHANGERATE_API_KEY не задан, инструмент курса валют отключён")
    return weather_client
