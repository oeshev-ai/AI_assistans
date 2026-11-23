# AI Assistants

Telegram-бот с несколькими персонами и темами. Диалог стримится, модель сама вызывает инструменты: погода, валюты, картинки, видео, расшифровка и озвучка. Основной провайдер — Polza.ai, запасной — ProxyAPI, ровно один раз на запрос.

Стек: Python 3.11, aiogram 3, Redis 7, httpx, arq, Prometheus, Sentry, Docker Compose.

## Что умеет бот

- Текстовый диалог со стримингом и function calling. История режется до `MAX_HISTORY_LENGTH` (по умолчанию 20) и не обрывается посередине tool-раунда.
- Несколько персон на чат: имя, иконка, свой системный промпт, текстовая, vision- и image-модель, размер картинки, голос. К общим правилам инструментов промпт персоны дописывается при запросе, в Redis хранится только текст персоны.
- Темы диалога внутри персоны. У каждой темы своя история.
- Каталог готовых персон в `/settings`: Помощник, Сеньор-разработчик, Копирайтер, Бизнес-аналитик, Репетитор, Переводчик, Художник, Быстрый помощник, Аналитик документов, Сказочник.
- Голос. Входящее голосовое (до 10 минут и 20 МБ) распознаётся и уходит в диалог как текст. Ответ озвучивается по режиму персоны: `off`, `auto` (только если пользователь прислал голос), `always`.
- Фото и PDF разбирает Vision. Альбом сначала смотрит дешёвый роутер и решает: правка, новая картинка, разбор или один уточняющий вопрос.
- Генерация и правка картинок, короткий ролик, расшифровка аудио и видео файлом `transcript.txt`.
- Файл больше лимита Telegram (~20 МБ) уходит по одноразовой HTTPS-ссылке в Object Storage, затем бот сам делает STT.
- Погода и курсы валют модель не выдумывает: для них есть отдельные инструменты.

Команды: `/start`, `/help`, `/cancel`, `/settings`. Внизу две кнопки: «Персоны» и «Темы». Остальное — inline-кнопки.

## Как устроен код

```text
app/
  bot/          aiogram: handlers/ по доменам, screens/ — клавиатуры и тексты
  llm/          клиент Polza и ProxyAPI, модели, картинки, аудио, видео, tools
  media/        загрузка больших файлов, ffmpeg, проверка сигнатуры файла
  storage/      персоны, темы, пресеты, сессии upload, S3
  infra/        бюджет, лимиты, breaker, метрики, health, очередь arq
bot.py          тонкая обёртка, то же что python -m app.bot
worker.py       точка входа arq: worker.WorkerSettings
```

Запрос идёт так: хендлер → активная персона и тема → `AssistAIClient` → провайдер. Картинки, TTS и STT при `ARQ_ENABLED=1` ставит в очередь процесс бота, считает их воркер. Чат остаётся в процессе бота.

Диалоги живут в `TopicHistoryStorage` (`TOPIC_TTL_DAYS`, по умолчанию 30). Обрезка ленты — `app/storage/message_trim.py`. `RedisConfig` в `app/storage/redis_config.py` задаёт только `REDIS_URL` для профилей ассистентов. `create_topic_storage` и `create_assistant_storage` в production без живого Redis завершают процесс; вне production пустой или недоступный URL оставляет память процесса.

Снятые id моделей (`gpt-image-1.5`, `qwen3.8-max`, `eleven-multilingual-v2` и остальные из таблиц в `app/llm/models_config.py`) при чтении профиля из Redis переписываются на актуальные и сразу сохраняются обратно. Следующее чтение уже не логирует алиас. Выключить подмену: `IMAGE_LEGACY_ALIASES=0`, `TEXT_LEGACY_ALIASES=0`, `TTS_LEGACY_ALIASES=0`.

## Провайдеры

Нужны оба ключа: `POLZA_API_KEY` и `PROXYAPI_API_KEY`. По умолчанию текст, картинки, аудио и видео идут на Polza. ProxyAPI — один запасной вызов, если основной недоступен: таймаут, 5xx, открытый circuit breaker, 429, нет баланса. Коды 400, 401, 403, 404, 413 и внутренние `guard_*` не повторяются и на fallback не уходят. Исчерпание `POLZA_TOTAL_WAIT` тоже не уходит на запасной провайдер: задача уже принята, бот забирает тот же результат позже.

Вернуть канал на ProxyAPI: соответствующий `*_PROVIDER=proxyapi`.

| Канал | Куда по умолчанию | Запасной |
|---|---|---|
| `TEXT_PROVIDER` | Polza `POST /api/v1/chat/completions` | ProxyAPI |
| `IMAGE_PROVIDER` | Polza `POST /v1/media` | ProxyAPI `/images/*` |
| `AUDIO_PROVIDER` | Polza `/api/v1/audio/*` (STT и каталожный TTS) | ProxyAPI |
| `VIDEO_PROVIDER` | Polza `/v1/media`, модель Seedance | ProxyAPI |
| `TTS_PROVIDER` | `polza`, `proxyapi` или `speechkit` | маршрут audio, если это не SpeechKit |

SpeechKit (`TTS_PROVIDER=speechkit`) ходит в Yandex TTS v3 и в эту схему fallback не входит. Нужны `YC_SPEECHKIT_API_KEY` и `YC_FOLDER_ID`.

Картинки на Polza:

- `openai/gpt-image-2.6-flare` — `aspect_ratio` и пиксельный `size`. Поле `image_resolution` этот id не принимает (HTTP 400). Точная сетка нестандартной пропорции может отличаться от запрошенной. Прежнее поведение ProxyAPI: `IMAGE_PROVIDER=proxyapi`.
- FLUX, Qwen и Nano Banana — `aspect_ratio` и `image_resolution`, без `quality`.
- `qwen/qwen-image-2.1-pro` на Polza уходит как `qwen/image-2.1`.
- Референсы — base64. Статус асинхронной задачи — только `GET /v1/media/{id}`.
- Апстрим картинок пиннится `POLZA_UPSTREAM_ONLY` (по умолчанию `OpenAI`). Потолок цены — `POLZA_MAX_PRICE_IMAGE_RUB` (0 снимает потолок). Для token pricing поле `max_price.image` не отправляется: Polza отвечает 400.
- Бот скачивает результат и отдаёт в Telegram байтами. Ссылка провайдера в чат не уходит.

Ожидание картинки — `POLZA_TOTAL_WAIT` секунд (240), опрос раз в `POLZA_POLL_INTERVAL` (4). Видео ждёт не меньше `max(POLZA_TOTAL_WAIT, 600)`.

## Каталог моделей

Кнопки бота берут списки из `app/llm/models_config.py`, каталог провайдера на каждое нажатие не запрашивается. Пустой выбор в профиле — модель из окружения.

Текст: `deepseek/deepseek-v4.1-flash-1017` (дефолт `TEXT_MODEL`), `google/gemini-3.8-flash`, `google/gemini-4-argon`, `anthropic/claude-sonnet-5-5` (на Polza id `anthropic/claude-sonnet-5.5`), `openai/gpt-6-luna`, `openai/gpt-6.1-terra`, `openai/gpt-6-sol`, `qwen/qwen4-max`.

Картинки: `openai/gpt-image-2.6-flare` (дефолт генерации и правки), `black-forest-labs/flux-3-image-klein`, `black-forest-labs/flux-3-image`, `qwen/qwen-image-2.1-pro`, `google/gemini-nano-banana-2.1`.

Озвучка: `openai/gpt-4o-mini-tts` (голоса alloy, verse, nova, shimmer) и `yandex/speechkit`. STT по умолчанию — `openai/whisper-large-v3`. Видео — `bytedance/seedance-2.0`, на Polza это `bytedance/seedance-2`.

Роутер фото по умолчанию — `PHOTO_ROUTER_MODEL=deepseek/deepseek-v4.1-flash-1017` с выключенным рассуждением (`effort=none`). Он возвращает JSON: правка, новая картинка, разбор или вопрос. Порог уверенности `PHOTO_ROUTER_CONFIDENCE=0.55`. Уточнений не больше `PHOTO_MAX_CLARIFY_ROUNDS` (2). После правки или генерации персона коротко подтверждает результат (`PHOTO_CONFIRM_MODE=A`); `C` отвечает шаблоном и модель не вызывает. `PHOTO_ROUTER_FAST_PATH=1` пропускает роутер и сразу правит фото — в production бот пишет предупреждение, для обычной работы оставьте 0.

Альбом собирается в процессе бота (`ALBUM_DEBOUNCE_SECONDS=1.2`), не больше `MAX_ALBUM_ITEMS` (5) и `MAX_ALBUM_TOTAL_MB` (20). Метаданные фото дублируются в Redis, чтобы второй процесс мог дописать тот же `media_group_id`. Ожидание подписи к фото живёт `PHOTO_PENDING_TTL_SECONDS` (900).

## Инструменты

Модель вызывает их сама. Общие правила дописываются к промпту каждой персоны.

Погода (нужен `OPENWEATHER_API_KEY`): `get_weather_by_city`, `get_weather_by_coords`, `get_forecast_by_city`, `get_forecast_by_coords`, `get_air_pollution`, `compare_cities`, `format_notification`.

Валюты (нужен `EXCHANGERATE_API_KEY`): `convert_currency`.

Медиа: `generate_image`, `edit_image`, `generate_video`, `transcribe_media`, `synthesize_speech`, `request_large_media_upload`.

Без ключа погоды или валют соответствующий инструмент не регистрируется. Без `YC_*` и `UPLOAD_PUBLIC_BASE_URL` ссылка на большую загрузку не выдаётся, остальной бот работает.

Загрузка больших файлов: одноразовая страница на health-порту, токен в HttpOnly-cookie, MIME проверяется и по расширению, и по сигнатуре. Объект лежит в Yandex Object Storage (`YC_BUCKET_NAME`, ключи, `YC_S3_ENDPOINT`). После загрузки бот скачивает объект и делает STT. Параллельно не больше `UPLOAD_MAX_CONCURRENT` (2) таких задач: файл целиком держится в памяти. Лимит тела — `UPLOAD_MAX_MB` (512). За nginx с TLS нужен `TRUST_PROXY=1` (в Compose уже стоит).

## Деньги и лимиты

Единица учёта — условный USD: `RUB_PER_USD` по умолчанию 304, это курс, к которому раньше приводили расход ProxyAPI. Фактические рубли Polza (`usage.cost_rub` или `usage.cost`) делятся на этот курс. Если рублей в ответе нет, берётся оценка по токенам, длительности или символам; для Polza она ещё умножается на `POLZA_COST_MULTIPLIER` (0.39). ProxyAPI рублей не отдаёт: чат считается по токенам или по символам, аудио — по оценке.

Перед платным вызовом сумма резервируется в Redis (`check_budget`). После ответа резерв закрывается фактической ценой, остаток снимается. Вложенный вызов инструмента не трогает резерв родительского хода. В production, если Redis не смог записать резерв, запрос отклоняется.

`0` в `LLM_DAILY_BUDGET_USD` и `LLM_USER_DAILY_BUDGET_USD` означает «без потолка» только в development. Compose ставит `ENVIRONMENT=production`, и процесс не стартует, пока не задан хотя бы один из этих потолков.

Отдельные скользящие окна на пользователя, счётчики в Redis:

| Операция | Лимит | Окно |
|---|---:|---:|
| Текст и чат | 10 | 60 с |
| Картинки | 5 | 1 ч |
| STT, TTS, видео | 10 | 1 ч |

При сбое Redis в production лимитер тоже отказывает, а не переходит на память процесса.

## Что обязательно в production

Compose сам выставляет `ENVIRONMENT=production`, `REDIS_URL` на сервис `redis`, `TRUST_PROXY=1`, health и metrics на `0.0.0.0` внутри контейнера. Наружу порты публикуются только на `127.0.0.1`.

Бот завершается с `SystemExit`, если в production:

- пустой `TELEGRAM_ALLOWED_USER_IDS`;
- оба дневных бюджета равны 0;
- нет `REDIS_URL` или Redis не отвечает (FSM, общий клиент бюджетов и лимитов, темы, персоны);
- `ARQ_ENABLED=1`, а пул очереди не поднялся — в процесс бота задачи не переносятся.

Локально без Docker пустой allowlist пускает всех, бюджеты могут быть нулевыми, а недоступный Redis заменяется памятью процесса. Эта память не переживает рестарт и не общая между ботом и воркером.

Allowlist проверяет сообщения и callback-кнопки. Чужому пользователю бот отвечает, что доступ только у авторизованных.

Фильтр фраз вроде «игнорируй предыдущие инструкции» — эвристика перед отправкой текста в модель, не граница доступа. Длины ограничены: сообщение 4000 символов, промпт картинки 3000.

Пароли в URL вырезаются из логов. Sentry, если задан `SENTRY_DSN`, не получает id пользователя и чата.

## Запуск

Минимум в `.env` (скопируйте `.env.example`):

```text
TELEGRAM_BOT_TOKEN
POLZA_API_KEY
PROXYAPI_API_KEY
```

Для Compose дополнительно:

```text
TELEGRAM_ALLOWED_USER_IDS=<id из @userinfobot>
LLM_USER_DAILY_BUDGET_USD=1.5
```

Поднять Redis и бота:

```bash
docker compose up -d --build
```

Проверка: `GET http://127.0.0.1:8082/live` отвечает, что процесс жив. `GET /ready` и `GET /health` пингуют Redis и провайдеров и отдают 503, если зависимость недоступна. Метрики Prometheus — `http://127.0.0.1:8083/metrics`. Порты меняются через `HEALTH_PORT` и `METRICS_PORT`.

Очередь картинок, TTS и STT (в `.env` ещё `ARQ_ENABLED=1`):

```bash
docker compose --profile arq up -d --build
```

Воркер — `arq worker.WorkerSettings`. Он выполняет `generate_image_task`, `edit_image_task`, `synthesize_speech_task`, `transcribe_audio_task`. `ARQ_JOB_TIMEOUT` не может быть меньше `180 + POLZA_TOTAL_WAIT + 30` (создание и опрос картинки), иначе воркер не стартует. По умолчанию таймаут задачи 450 с, одна попытка на задачу, одновременно до `ARQ_MAX_JOBS` (10).

Снимок Redis (профиль `backup`, файлы в `./backups/redis`, хранятся 7 дней):

```bash
docker compose --profile backup run --rm redis-backup
```

Без Docker:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
docker compose up -d redis
python -m app.bot
```

`python bot.py` делает то же самое. Локальный Redis без пароля: `REDIS_URL=redis://127.0.0.1:6379/0`. Пустой `REDIS_URL` вне production оставляет персоны и темы в памяти процесса.

Тесты и статика (так же, как CI на `main`):

```bash
ruff check .
ruff format --check .
mypy . --ignore-missing-imports
pytest tests/ -v
```

Прод-зависимости без pytest, ruff и mypy — `requirements-prod.txt`. Образ ставит их и ffmpeg, процесс внутри контейнера идёт не от root.

## Деплой

CI на push и pull request в `main`: ruff, format, mypy, pytest, установка `requirements-prod.txt`, сборка Docker-образа.

После зелёного CI на push в `main` workflow Deploy публикует `ghcr.io/oeshev-ai/ai_assistans:latest` и тег по SHA. Без `APP_IMAGE` Compose собирает локальный тег `ai-assistans:local`. SSH-шаг в workflow по умолчанию выключен.

### Чеклист перед включением Deploy over SSH

VPS: пользователь `deploy`, каталог `/home/deploy/AI_assistans` (не `/opt/vpd06`).

1. На `main` есть актуальный `docker-compose.yml` с `image: ${APP_IMAGE:-…}` (не только `build: .`). На сервере: `git pull` до этого коммита.
2. В `/home/deploy/AI_assistans/.env`:
   ```text
   APP_IMAGE=ghcr.io/oeshev-ai/ai_assistans:latest
   ```
3. Под `deploy`: `docker login ghcr.io` (PAT с `read:packages`), если пакет private.
4. Вручную: `cd /home/deploy/AI_assistans && docker compose pull && docker compose up -d` — контейнер бота должен перейти на образ `ghcr.io/oeshev-ai/…`, не `ai_assistans-bot`.
5. Секреты репозитория: `SSH_HOST`, `SSH_USER` (`deploy`), `SSH_PRIVATE_KEY` (отдельный ключ для Actions, pubkey в `authorized_keys` у `deploy`).
6. В `.github/workflows/deploy.yml` раскомментировать шаг `Deploy over SSH`. При `ARQ_ENABLED=1` в script использовать `--profile arq` для `pull` и `up`.

Redis в Compose: AOF, потолок `REDIS_MAXMEMORY` (по умолчанию 256mb), политика `volatile-ttl`. У тем, персон, бюджета, лимитов и буферов очереди есть срок жизни. Когда память кончается, Redis снимает ключ с самым коротким остатком TTL: сначала окна лимитов и временные payload, потом давно не открытые темы. Активный чат продлевает TTL при обращении и освобождается последним. Запись истории при этом не останавливается целиком. Пароль — `REDIS_PASSWORD`; пустой пароль не подставляется в URL, иначе клиент arq ломает AUTH.

## Метрики

Счётчики, которые имеет смысл смотреть:

- `llm_spend_usd_total{kind}` — расход по виду вызова (chat, image, stt, tts, video).
- `llm_cost_usd_total{kind,provider,model,source}` — тот же расход с разрезом. `source`: `cost_rub`, `usage`, `table`, `flat`.
- `llm_daily_budget_remaining_usd` — остаток глобального дневного потолка, если он задан.
- `provider_fallback_total{from,to,reason}` — уход на запасной провайдер.
- `api_calls_total`, `api_latency_seconds`, `errors_total` — внешние вызовы, метка `service` равна id провайдера (`polza` или `proxyapi`).
- `message_processing_seconds`, `active_users`, `redis_memory_bytes`.

Логи — JSON structlog. Секреты и пароль в Redis URL в них не пишутся.

## Чего нет в git

`.env` не коммитится. Каталог `app/docs/` тоже в `.gitignore`: локальные контракты Polza, заметки по приватности и короткий runbook остаются на машине, в клон репозитория не попадают. Поведение, от которого зависит запуск, описано здесь и в комментариях `.env.example`.

## Контакт

Telegram: [@OEShevchenko](https://t.me/OEShevchenko)  
Email: oeshevip@yandex.ru
