# Telegram-бот с LLM через ProxyAPI

Асинхронный бот на aiogram 3: диалог со стримингом, персоны и темы в Redis,
генерация картинок, анализ фото/PDF, голос (STT/TTS), погода и курсы валют
через Function Calling.

LLM и изображения идут через [ProxyAPI](https://proxyapi.ru/).

## Состав

| Файл | Назначение |
|------|------------|
| `bot.py` | Telegram-бот: меню, FSM-мастера, стриминг, Vision, голос |
| `Assist_ai.py` | Клиент ProxyAPI: chat + tools + картинки + STT/TTS |
| `assistant_storage.py` | Профили персон (промпт, модели, голос) в Redis или памяти |
| `topic_history.py` | Несколько тем диалога на персону |
| `history_storage.py` | Общий конфиг Redis и утилиты истории |
| `vision.py` | Скачивание вложений, multimodal-контент, PDF → страницы |
| `models_config.py` | Списки текстовых и image-моделей для кнопок настроек |
| `OpenWeatherAPI.py` | OpenWeatherMap: погода, прогноз, воздух, сравнение городов |
| `ExchangerateAPI.py` | Конвертация валют через `api.exchangerate.host` |

## Требования

- Python 3.11+ (рекомендуется 3.11.9)
- Ключи API (см. `.env.example`)
- Для продакшена: Docker и Docker Compose (бот + Redis)

## Установка (локально)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Заполните `.env` реальными ключами. Файл `.env` в git не попадает.

Для запуска только бота без dev-зависимостей достаточно `requirements-prod.txt`.

## Переменные окружения

| Переменная | Для чего |
|------------|----------|
| `PROXYAPI_API_KEY` | Текст, tools, картинки, STT/TTS через ProxyAPI |
| `PROXYAPI_TRANSCRIBE_MODEL` | Модель распознавания речи (по умолчанию `openai/whisper-1`) |
| `PROXYAPI_TTS_MODEL` | Модель озвучки |
| `PROXYAPI_TTS_VOICE` | Голос TTS по умолчанию (`alloy` и др.) |
| `PROXYAPI_TTS_FORMAT` | Формат аудио (`opus` и др.) |
| `OPENWEATHER_API_KEY` | Инструменты погоды |
| `EXCHANGERATE_API_KEY` | Конвертация валют |
| `TELEGRAM_BOT_TOKEN` | Токен от @BotFather |
| `REDIS_URL` | История и профили. Пусто — память одного процесса |
| `HISTORY_TTL_DAYS` | TTL вспомогательного хранилища (по умолчанию 7) |
| `TOPIC_TTL_DAYS` | Через сколько дней Redis удаляет неактивную тему (30) |
| `MAX_HISTORY_LENGTH` | Сколько последних сообщений хранить в одной теме |

Без ключа погоды или курса соответствующий инструмент не регистрируется.
Без `PROXYAPI_API_KEY` клиент LLM не стартует. Без `TELEGRAM_BOT_TOKEN` не
стартует бот. Если Redis недоступен, профили и темы остаются в памяти процесса.

## Docker

```powershell
copy .env.example .env
# заполните TELEGRAM_BOT_TOKEN и PROXYAPI_API_KEY
docker compose up -d --build
```

Compose поднимает Redis (`redis:7-alpine`) и бота. `REDIS_URL` в контейнере
подставляется как `redis://redis:6379/0`. Образ собирается из `Dockerfile`
(Python 3.11-slim, зависимости из `requirements-prod.txt`).

После изменений кода: `docker compose up -d --build`.

## Быстрый старт

### Telegram-бот

```powershell
python bot.py
```

Нужен `TELEGRAM_BOT_TOKEN`. Один `AssistAIClient` создаётся при старте и
закрывается при остановке. Темы и персоны — в Redis (`docker compose up`) или
в памяти без `REDIS_URL`. В модель уходит активная тема и system-промпт
активной персоны (`compose_system_prompt`: текст пользователя + общие правила
tools/Vision).

**Что умеет бот**

- Текст со стримингом ответа
- Персоны: имя, системный промпт, текстовая и image-модель, размер картинки, голос
- Темы диалога с отдельной историей на персону
- Генерация изображений (`/image` и tool `generate_image`; семейства OpenAI / FLUX / Qwen / Seedream / Recraft)
- Vision: фото и PDF (первые страницы) через multimodal chat
- Голос: STT входящих голосовых; TTS ответа по режиму персоны (`выкл` / `авто` / `всегда`)
- Погода и валюты через Function Calling

Нижнее меню: **Ассистенты**, **Темы**, **Настройки**. Модели и голос правятся
в настройках или командами ниже.

**Команды:** `/help`, `/assistants`, `/create_assistant`, `/delete_assistant`,
`/model`, `/image_model`, `/voice`, `/new_topic`, `/topics`, `/switch`,
`/clear_topic`, `/image <описание>`, `/cancel`.

### Демо клиента LLM

```powershell
python Assist_ai.py
```

Проверяет Function Calling (погода, валюта) и генерацию изображения.

### Только погода

```powershell
python OpenWeatherAPI.py
```

## AssistAIClient

```python
import asyncio
from Assist_ai import AssistAIClient


async def main() -> None:
    async with AssistAIClient() as client:
        answer = await client.chat_completion(
            [{"role": "user", "content": "Какая погода в Казани?"}]
        )
        print(answer)

        async for delta in client.stream_chat(
            [{"role": "user", "content": "17 * 23?"}],
            model="deepseek/deepseek-v4.1-flash",
        ):
            if delta.kind == "content":
                print(delta.text, end="", flush=True)


asyncio.run(main())
```

Дефолты клиента (если в профиле модели не заданы): текст `qwen/qwen3.8-flash`,
картинки `openai/gpt-image-1-mini`. Актуальные списки для кнопок бота —
в `models_config.py` (каталог [ProxyAPI Models](https://proxyapi.ru/models/browse)).

### Инструменты модели

- `get_weather_by_city` / `get_weather_by_coords` — текущая погода и суточный прогноз
- `get_forecast_by_city` / `get_forecast_by_coords` — прогноз с шагом 3 часа
- `get_air_pollution` — качество воздуха
- `compare_cities` — сравнение двух городов
- `format_notification` — готовый текст уведомления
- `convert_currency` — конвертация суммы
- `generate_image` — генерация изображения (модель и размер из активной персоны)

Инструменты вызывает модель, когда они нужны для ответа.

## Тесты и CI

```powershell
pytest tests/ -v
ruff check .
ruff format --check .
mypy . --ignore-missing-imports
```

GitHub Actions (`.github/workflows/ci.yml`): ruff, mypy, pytest на Python 3.11.

## Зависимости

См. `requirements.txt` (dev) и `requirements-prod.txt` (Docker):

- `httpx`, `pydantic`, `python-dotenv`, `tenacity`
- `aiogram` 3.x, `redis` 5.x, `openai` (совместимость окружения)
- `PyMuPDF` — разбор PDF для Vision

Запросы к ProxyAPI, погоде и курсу идут через `httpx`. Telegram-транспорт у
aiogram — `aiohttp`.
