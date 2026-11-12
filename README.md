# AI Assistants — мультимодальный Telegram-бот

> Telegram-бот с персональными AI-ассистентами: стриминг диалога, картинки, голос, Vision и инструменты (погода, валюты, медиа) через ProxyAPI.

## Проблема

Нужен был рабочий ассистент в Telegram «из коробки»: не один чат с моделью, а несколько персон и тем с историей, плюс картинки, голос и разбор фото/PDF. Готовые боты либо слишком простые, либо не живут в проде без лимитов, логов и нормального деплоя. Отдельно мешал лимит Telegram на файлы ~20 МБ — длинные аудио/видео нельзя было нормально расшифровать.

## Решение

Собран production-ready бот на aiogram 3 и едином клиенте LLM (ProxyAPI): персоны с промптами и моделями, темы в Redis, function calling, Vision, STT/TTS, генерация и правка картинок. Тяжёлые задачи выносятся в arq-worker; большие файлы принимаются по одноразовой HTTPS-ссылке в Object Storage. Наблюдаемость — JSON-логи, Prometheus, health, rate limits, circuit breaker, бюджеты LLM.

## Что внутри

- **Технологии:** Python 3.11, aiogram 3, Redis, ProxyAPI (LLM/Vision/STT/картинки), Yandex SpeechKit TTS, Yandex Object Storage, arq, Prometheus, Sentry, Docker Compose, GitHub Actions
- **Архитектура:** пакет `app/` — `bot` (`handlers/` и `screens/` по доменам) → `llm` / `media` → `storage` + `infra`; опционально очередь arq для image/TTS/STT
- **Деплой:** VPS + Docker Compose + nginx (health/upload на localhost), CI: ruff / mypy / pytest

**Возможности:** стриминг ответов, каталог персон, темы диалога, Vision (фото/PDF), голос, generate/edit image и video, транскрипты файлом, погода и валюты через tools.

## Как запустить

```bash
git clone https://github.com/oeshev-ai/AI_assistans.git
cd AI_assistans
cp .env.example .env   # минимум: TELEGRAM_BOT_TOKEN, PROXYAPI_API_KEY
docker compose up -d --build
```

Локально без Docker: `python -m venv .venv`, активировать, `pip install -r requirements.txt`, заполнить `.env`, `python -m app.bot` (для истории удобен `docker compose up -d redis`).

Тесты: `pytest tests/ -v`.

## Результат

- Один бот закрывает диалог, персон, медиа и tools вместо набора разрозненных скриптов
- История и профили переживают рестарт (Redis); без Redis — in-memory fallback
- Файлы > ~20 МБ обрабатываются через upload → Object Storage → авто-STT
- Прод-контур: лимиты, circuit breaker, метрики, health, CI на каждый push

## Контакт

Telegram: [@OEShevchenko](https://t.me/OEShevchenko)  
Email: oeshevip@yandex.ru
