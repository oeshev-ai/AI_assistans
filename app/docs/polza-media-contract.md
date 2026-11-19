# Polza Media contract (G0 probe)

Model: `openai/gpt-image-2.6-flare`  
Base: `https://polza.ai/api`  
Источник: живой прогон `scripts/probe_polza.py` (`artifacts/polza_probe/20261119T043249Z/`).

## Поле → наблюдение

| поле / вопрос | наблюдение |
|---|---|
| `image_resolution` | **не поддерживается** для flare: HTTP 400 `BAD_REQUEST` («не поддерживает параметр image_resolution») |
| `resolution` | то же: 400 (discovery) |
| размер для flare | `aspect_ratio` + пиксельный `size` (`1536x1024` и т.п.); `size=1K` тоже принимается |
| `aspect_ratio=3:2` + `size=1536x1024` | результат **1536×1024** (IHDR); sync `status=completed` за ~6.8 с |
| `images[].type=base64` | **принят**; edit completed, результат 1536×1024; `cost_rub≈2.47` |
| `quality=xhigh` | принят, completed; `output_tokens=3122`, `cost_rub≈11.12` |
| `quality=max` | принят (async create → `pending`, без 400) |
| `usage.cost_rub` | есть в **синхронном** completed-ответе POST; алиас `cost` совпадает |
| неизвестное `complexity` | **молча игнорируется** (не 400, `warnings[]` пуст) |
| форма `data` | **массив** `[{ "url": "https://s3.polza.ai/..." }]`, не объект |
| polling | рабочий путь **`GET /v1/media/{id}`**; `/v2/media/{id}` → **404** `NOT_FOUND` |
| async-окно | без `async`: часто completed в POST (~6–13 с); с `async:true`: pending → processing → completed, ~3 poll / ~8.5 с |
| CDN | файлы на `s3.polza.ai`, 7 дней; полный download с этой сети может быть медленным — размеры читаются из PNG IHDR |

## Маппинг размера (для G3)

Для `openai/gpt-image-2.6-flare` на `/v1/media`:

1. `fit_image_size` / исходник → ближайший `aspect_ratio` из enum Polza.
2. Пиксели передавать полем `size` (`WIDTHxHEIGHT`), **не** `image_resolution`.
3. `image_resolution` / `resolution` не отправлять — каталог модели отвечает 400.
4. `complexity` и `background` не отправлять (`complexity` молча игнорируется).
5. Ответ: брать первый элемент `data[]` → `url` или `b64_json`; `usage` целиком в учёт.

## Фикстуры

- `tests/fixtures/polza/generate_completed.json`
- `tests/fixtures/polza/edit_completed.json`
- `tests/fixtures/polza/async_pending.json`
- `tests/fixtures/polza/error_bad_request.json`

## Стоп-условие

base64 принят — можно строить адаптер на `MediaFileDto.type=base64` без S3.

## Реализация (G3–G8)

- Адаптер: `app/llm/polza_media.py` (`generate`, `polza_size_fields`, `build_media_body`, `map_media_error`); wiring и fallback — `app/llm/assist_ai.py`.
- Размер: `fit_image_size` → `polza_size_fields` → `aspect_ratio` (ближайшее из `POLZA_ASPECTS`) + `size=WxH`; `auto` → `aspect_ratio: auto`. `image_resolution_tier` оставлен только как справка.
- Тело: `{model, input{prompt, aspect_ratio, size, quality, output_format, max_images, images[base64]}, user, provider{only, allow_fallbacks, max_price.image}}`; `complexity`, `background`, `require_parameters`, `data_residency` не отправляются.
- Ошибки: `error.code` → `AssistAIError.error_code`; `trace_id`, `metadata.reason`, `metadata.provider_name` — только в лог (`polza_media_error`), `warnings[]` — `polza_media_warnings`.
- Fallback (один раз): таймаут, 5xx, `SERVICE_UNAVAILABLE`, `BAD_GATEWAY`, `REQUEST_TIMEOUT`, `INTERNAL_ERROR`, open-circuit, исчерпание `POLZA_TOTAL_WAIT`, 429 после двух повторов, `INSUFFICIENT_BALANCE` (+ `provider_balance_exhausted`). Без fallback: `BAD_REQUEST`, `UNAUTHORIZED`, `FORBIDDEN`, `guard_*`, `NOT_FOUND`, `PAYLOAD_TOO_LARGE`.
- Лимит референсов: 16 (`POLZA_MAX_REFERENCE_IMAGES`).
