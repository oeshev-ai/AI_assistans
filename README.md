# AI Assistants — мультимодальный Telegram-бот

> Telegram-бот с персональными AI-ассистентами: стриминг диалога, картинки, голос, Vision и инструменты (погода, валюты, медиа) через ProxyAPI.

## Проблема

Нужен был рабочий ассистент в Telegram «из коробки»: не один чат с моделью, а несколько персон и тем с историей, плюс картинки, голос и разбор фото/PDF. Готовые боты либо слишком простые, либо не живут в проде без лимитов, логов и нормального деплоя. Отдельно мешал лимит Telegram на файлы ~20 МБ — длинные аудио/видео нельзя было нормально расшифровать.

## Решение

Собран production-ready бот на aiogram 3 и едином клиенте LLM (ProxyAPI): персоны с промптами и моделями, темы в Redis, function calling, Vision, STT/TTS, генерация и правка картинок. Тяжёлые задачи опционально уходят в arq-worker; большие файлы — по одноразовой HTTPS-ссылке в Object Storage. Наблюдаемость — JSON-логи, Prometheus, health, rate limits, circuit breaker, бюджеты LLM.

## Что внутри

- **Технологии:** Python 3.11, aiogram 3, Redis, ProxyAPI (LLM/Vision/STT/картинки), Yandex SpeechKit TTS, Yandex Object Storage, arq, Prometheus, Sentry, Docker Compose, GitHub Actions
- **Архитектура:** пакет `app/` — `bot` (`handlers/` и `screens/` по доменам) → `llm` / `media` → `storage` + `infra`; опционально очередь arq для image/TTS/STT
- **Деплой:** VPS + Docker Compose + nginx (health/upload на localhost), CI: ruff / mypy / pytest

**Возможности:** стриминг ответов, каталог персон, темы диалога, Vision (фото/PDF), голос, generate/edit image и video, транскрипты файлом, погода и валюты через tools. Генерация и правки картинок идут через GPT Image 2.6 Flare, быстрые черновики — FLUX klein. Постер 1024×1536 на high у Flare занимает около 11.5 с. Фото и альбомы сначала разбирает отдельная дешёвая модель и возвращает JSON: правка, новая картинка, анализ или уточняющий вопрос без кнопок.

Фактическая стоимость GPT Image на ProxyAPI (304 ₽/$):

| запрос | size | quality | $ | ₽ |
|---|---|---|---:|---:|
| генерация | 1024×1024 | low | 0.02476 | 7.53 |
| правка | 1536×1024 | high | 0.053003 | 16.11 |
| правка, size=auto, вход 1024² | — | low | 0.037162 | 11.30 |
| правка, size=auto, вход 4032×3024 | — | low | 0.033416 | 10.16 |
| генерация афиши | 1024×1536 | high | 0.041905 | 12.74 |

## Как запустить

```bash
git clone https://github.com/oeshev-ai/AI_assistans.git
cd AI_assistans
cp .env.example .env   # минимум: TELEGRAM_BOT_TOKEN, PROXYAPI_API_KEY
docker compose up -d --build
```

По умолчанию поднимаются `redis` + `bot`; image/TTS/STT выполняются **в процессе бота** (`ARQ_ENABLED=0`). Очередь arq — опционально:

```bash
# в .env: ARQ_ENABLED=1
docker compose --profile arq up -d --build
# worker: arq worker.WorkerSettings
```

Локально без Docker: `python -m venv .venv`, активировать, `pip install -r requirements.txt`, заполнить `.env`, `python -m app.bot` (для истории удобен `docker compose up -d redis`).

Тесты: `pytest tests/ -v`.

## Результат

- Один бот закрывает диалог, персон, медиа и tools вместо набора разрозненных скриптов
- История и профили переживают рестарт (Redis, `maxmemory-policy noeviction`); без Redis — in-memory fallback
- Файлы > ~20 МБ обрабатываются через upload → Object Storage → авто-STT
- Прод-контур: allowlist (`TELEGRAM_ALLOWED_USER_IDS`), опциональный `REDIS_PASSWORD`, лимиты, circuit breaker, метрики, health, CI
- Альбомы фото буферизуются в процессе бота — один экземпляр polling (не несколько реплик без sticky routing)

## Контакт

Telegram: [@OEShevchenko](https://t.me/OEShevchenko)  
Email: oeshevip@yandex.ru
