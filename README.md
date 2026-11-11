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
| `logging_setup.py` | JSON structured logging (structlog) |
| `metrics.py` | Prometheus-метрики |
| `health_server.py` | HTTP `/health` |
| `rate_limiter.py` | Лимиты LLM/картинок (Redis или память) |
| `security.py` | Санитизация prompt injection и валидация длины |
| `circuit_breaker.py` | Async circuit breaker для внешних API |
| `job_queue.py` | Очередь тяжёлых задач (arq) с inline fallback |
| `worker.py` | arq worker: image / TTS / STT |
| `cost_tracker.py` | Учёт расходов и дневные бюджеты LLM |
| `scripts/backup_redis.sh` | Бэкап Redis RDB (Linux/cron/compose) |
| `scripts/backup_redis.ps1` | Бэкап Redis RDB (Windows) |

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
| `PROXYAPI_API_KEY` | Текст, tools, картинки, STT через ProxyAPI |
| `PROXYAPI_TRANSCRIBE_MODEL` | Модель распознавания речи (по умолчанию `openai/whisper-large-v3`) |
| `TTS_PROVIDER` | Озвучка: `speechkit` (Yandex SpeechKit) или `proxyapi` |
| `YC_SPEECHKIT_API_KEY` | API-ключ сервисного аккаунта для SpeechKit (роль `ai.speechkit-tts.user`) |
| `YC_FOLDER_ID` | Каталог Yandex Cloud (`x-folder-id`) |
| `SPEECHKIT_VOICE` / `SPEECHKIT_ROLE` / `SPEECHKIT_SPEED` | Голос, амплуа и скорость SpeechKit |
| `PROXYAPI_TTS_MODEL` | Модель озвучки ProxyAPI (если `TTS_PROVIDER=proxyapi`) |
| `PROXYAPI_TTS_VOICE` | Голос ProxyAPI (`alloy` и др.) |
| `PROXYAPI_TTS_FORMAT` | Формат аудио ProxyAPI (`opus` и др.) |
| `OPENWEATHER_API_KEY` | Инструменты погоды |
| `EXCHANGERATE_API_KEY` | Конвертация валют |
| `TELEGRAM_BOT_TOKEN` | Токен от @BotFather |
| `REDIS_URL` | История, профили и FSM. Пусто — память одного процесса |
| `HISTORY_TTL_DAYS` | TTL вспомогательного хранилища (по умолчанию 7) |
| `TOPIC_TTL_DAYS` | Через сколько дней Redis удаляет неактивную тему (30) |
| `MAX_HISTORY_LENGTH` | Сколько последних сообщений хранить в одной теме |
| `METRICS_PORT` | Prometheus scrape (`/metrics`), по умолчанию `8083` |
| `HEALTH_PORT` | Health + страница загрузки больших файлов, по умолчанию `8082` (не `8081` — на VPS его занимает neurophoto) |
| `YC_BUCKET_NAME` / `YC_ACCESS_KEY_ID` / `YC_SECRET_ACCESS_KEY` | Yandex Object Storage для больших аудио/видео |
| `UPLOAD_PUBLIC_BASE_URL` | Публичный HTTPS до `HEALTH_PORT`, напр. `https://assist.mrischool.ru` |
| `UPLOAD_MAX_MB` | Лимит размера файла в бакет (по умолчанию 512) |
| `SENTRY_DSN` | Опциональный DSN для Sentry |
| `ENVIRONMENT` | Окружение для Sentry (`development` / `production`) |
| `APP_VERSION` | Версия релиза для Sentry |
| `ARQ_ENABLED` | `1` — тяжёлые задачи через arq worker |
| `LLM_DAILY_BUDGET_USD` | Глобальный дневной бюджет (0 = без лимита) |
| `LLM_USER_DAILY_BUDGET_USD` | Дневной бюджет на пользователя (0 = без лимита) |

Без ключа погоды или курса соответствующий инструмент не регистрируется.
Без `PROXYAPI_API_KEY` клиент LLM не стартует. Без `TELEGRAM_BOT_TOKEN` не
стартует бот. Если Redis недоступен, профили, темы и FSM остаются в памяти процесса.

Озвучка персоны по умолчанию в `.env.example` — **Yandex SpeechKit** (`TTS_PROVIDER=speechkit`):
API-ключ сервисного аккаунта с ролью `ai.speechkit-tts.user`, плюс `YC_FOLDER_ID`.
Формат — OGG Opus. STT по-прежнему через ProxyAPI whisper.

## Production observability

- JSON-логи через `structlog` (поля `user_id`, `chat_id`, `assistant_id`, `topic_id`)
- Prometheus: `http://127.0.0.1:8083/metrics`
- Health / upload: `http://127.0.0.1:8082/health` (Redis + ProxyAPI; наружу — через nginx)
- Rate limits: 10 LLM-запросов/мин и 5 картинок/час на пользователя
- Circuit breaker для LLM, погоды и валют
- Sentry при заданном `SENTRY_DSN`
- Учёт расходов LLM (`llm_spend_usd_total`) и дневные бюджеты

## Очереди (arq)

По умолчанию image/TTS/STT выполняются в процессе бота. Чтобы вынести их в worker:

```powershell
# .env
ARQ_ENABLED=1
REDIS_URL=redis://localhost:6379/0

# терминал 1
docker compose up -d redis
python bot.py

# терминал 2
arq worker.WorkerSettings
```

В Docker: `docker compose --profile arq up -d --build` (сервисы `bot` + `worker`).

## Backup Redis

```powershell
# Windows (нужен redis-cli в PATH)
.\scripts\backup_redis.ps1

# Linux / WSL / cron
REDIS_HOST=127.0.0.1 ./scripts/backup_redis.sh

# одноразовый контейнер
docker compose --profile backup run --rm redis-backup
```

Снимки пишутся в `backups/redis/` (каталог в `.gitignore`).

## Docker

```powershell
copy .env.example .env
# заполните TELEGRAM_BOT_TOKEN и PROXYAPI_API_KEY
docker compose up -d --build
```

Compose поднимает Redis (`redis:7-alpine`) и бота. `REDIS_URL` в контейнере
подставляется как `redis://redis:6379/0`. Порты `8083` (metrics) и `8082`
(health/upload) слушаются только на `127.0.0.1`, чтобы не конфликтовать с
neurophoto (`8081`) и не торчать в интернет без nginx. Образ — из `Dockerfile`
(Python 3.11-slim, `requirements-prod.txt`).

На VPS с `mrischool.ru`: DNS `assist` → IP сервера, nginx
`assist.mrischool.ru` → `http://127.0.0.1:8082`, в `.env`
`UPLOAD_PUBLIC_BASE_URL=https://assist.mrischool.ru`.
В nginx для этого `server` обязательно `client_max_body_size 512m;` —
файл теперь грузится на наш домен, а сервер сам кладёт его в Object Storage
(без браузерного CORS к Yandex; так надёжнее в Telegram WebView).

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
- Медиа-tools персоны: `generate_image`, `edit_image`, `generate_video`, `transcribe_media`, `synthesize_speech`, `request_large_media_upload`
- Транскрипт длинного аудио/видео уходит файлом `transcript.txt` (из видео сначала извлекается аудио через ffmpeg)
- Файлы > ~20 МБ: персона выдаёт одноразовую ссылку → загрузка в Yandex Object Storage → авто-STT (нужны `YC_*` и `UPLOAD_PUBLIC_BASE_URL`)
- Vision: фото и PDF через multimodal chat
- Голосовые сообщения: авто-STT как текст диалога; TTS ответа по `voice_mode`
- В настройках персоны можно выбрать TTS-модель (`yandex/speechkit`, ProxyAPI `openai/gpt-4o-mini-tts` / `elevenlabs/eleven-multilingual-v2`)
- Погода и валюты через Function Calling

Нижнее меню: **Ассистенты** и **Темы** (два ряда). Настройки — команда `/settings` в меню бота (☰).
Модели и голос правятся только в настройках.

**Команды:** `/help` (справка), `/cancel` (отмена мастера). `/start` — то же, что `/help`.

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
