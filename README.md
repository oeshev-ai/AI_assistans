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

### Провайдеры: роутинг, fallback и стоимость

По умолчанию всё идёт через ProxyAPI. Картинки (генерация и правки) можно отправлять на Polza.ai через `POST /v1/media`; ProxyAPI при этом остаётся страховкой. Текст, аудио и видео адресуются отдельными переменными, но остаются на ProxyAPI.

| провайдер | коэф. ₽ за $1 прайса OpenAI | правка 1536×1024 high | постер 1024×1536 high | генерация 1024² low |
|---|---:|---:|---:|---:|
| ProxyAPI | ~304 | 16.11 ₽ (замер) | 12.74 ₽ (замер) | 7.53 ₽ (замер) |
| Polza.ai | ~118.6 | ~6.29 ₽ (расчёт по прайсу) | ~4.97 ₽ (пересчёт) | ~2.94 ₽ (пересчёт) |

```text
 TEXT/AUDIO/VIDEO_PROVIDER ──► proxyapi
 IMAGE_PROVIDER ──► polza (/v1/media) ──ok──► результат
                        │ недоступен: таймаут, 5xx, SERVICE_UNAVAILABLE,
                        │ BAD_GATEWAY, REQUEST_TIMEOUT, breaker open,
                        │ исчерпан POLZA_TOTAL_WAIT, 429 после 2 повторов,
                        │ INSUFFICIENT_BALANCE (+ алерт provider_balance_exhausted)
                        ▼
              IMAGE_PROVIDER_FALLBACK (proxyapi) — ровно один раз
 400/401/403/404/413 и guard_* — без повторов и fallback
```

- Учёт: на Polza факт берётся из `usage.cost_rub / RUB_PER_USD`; если рублей нет — токены × `POLZA_COST_MULTIPLIER`. Единица учёта — прежний «условный USD»: **1 USD ≈ 304 ₽ реального расхода на ProxyAPI**.
- Метрики: `provider_fallback_total{from,to,reason}`, `llm_cost_usd_total{kind,provider,model,source}`; `llm_spend_usd_total{kind}` не менялась.
- **Потеря точных пикселей на Polza:** размер уходит как `aspect_ratio` (ближайшее из набора Polza) + `size` `WIDTHxHEIGHT`; для `gpt-image-2.6-flare` `image_resolution` не поддерживается (HTTP 400), итог проверен на 3:2 → 1536×1024, но для нестандартных пропорций сетка может отличаться. Вернуть прежнее поведение: `IMAGE_PROVIDER=proxyapi`.
- **CDN Polza хранит результат 7 дней**; бот сразу отдаёт картинку в Telegram, но ссылки в логах и истории протухнут.
- Апстрим пиннится (`POLZA_UPSTREAM_ONLY=OpenAI`, без fallback апстримов), потолок цены — `POLZA_MAX_PRICE_IMAGE_RUB`; контракт и наблюдения G0 — в `app/docs/polza-media-contract.md`.

## Как запустить

```bash
git clone https://github.com/oeshev-ai/AI_assistans.git
cd AI_assistans
cp .env.example .env   # минимум: TELEGRAM_BOT_TOKEN, PROXYAPI_API_KEY
docker compose up -d --build
```

На VPS после публикации в GHCR: в `.env` задать `APP_IMAGE=ghcr.io/<owner>/<repo>:latest`, затем `docker compose pull && docker compose up -d` (без `--build`).

По умолчанию поднимаются `redis` + `bot`; image/TTS/STT выполняются **в процессе бота** (`ARQ_ENABLED=0`). Очередь arq — опционально:

```bash
# в .env: ARQ_ENABLED=1
docker compose --profile arq up -d --build
# worker: arq worker.WorkerSettings
```

Локально без Docker: `python -m venv .venv`, активировать, `pip install -r requirements.txt`, заполнить `.env`, `python -m app.bot` (для истории удобен `docker compose up -d redis`).

Тесты: `pytest tests/ -v`.

Документы: [privacy/retention](app/docs/privacy-and-retention.md), [ops runbook](app/docs/ops-runbook.md), [Polza media](app/docs/polza-media-contract.md).

## Результат

- Один бот закрывает диалог, персон, медиа и tools вместо набора разрозненных скриптов
- История и профили переживают рестарт (Redis, `maxmemory-policy noeviction`); без Redis — in-memory fallback
- Файлы > ~20 МБ обрабатываются через upload → Object Storage → авто-STT
- Прод-контур: allowlist (`TELEGRAM_ALLOWED_USER_IDS`), опциональный `REDIS_PASSWORD`, лимиты, circuit breaker, метрики, health, CI
- Альбомы: debounce в процессе бота; метаданные фото дублируются в Redis (`shared_redis`), чтобы реплики могли склеить один `media_group_id`

## Контакт

Telegram: [@OEShevchenko](https://t.me/OEShevchenko)  
Email: oeshevip@yandex.ru
