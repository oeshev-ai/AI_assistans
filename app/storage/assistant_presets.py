"""Каталог готовых персон (пресетов) для создания профилей."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.llm.assist_ai import image_model_family
from app.llm.models_config import (
    AVAILABLE_IMAGE_MODELS,
    AVAILABLE_TEXT_MODELS,
    AVAILABLE_TTS_MODELS,
    ELEVENLABS_TTS_VOICES,
    OPENAI_TTS_VOICES,
    is_elevenlabs_tts_model,
)
from app.storage.assistant_storage import AssistantProfile
from app.storage.persona_icons import PERSONA_ICONS, normalize_icon


class AssistantPreset(BaseModel):
    """Шаблон персоны из каталога. Не хранится в Redis как профиль."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(min_length=1, max_length=16)
    icon: str = Field(min_length=1, max_length=8)
    name: str = Field(min_length=1, max_length=50)
    summary: str = Field(min_length=1, max_length=120)
    system_prompt: str = Field(min_length=10)
    text_model: str | None = None
    image_model: str | None = None
    image_size: str | None = None
    vision_model: str | None = None
    voice_mode: Literal["off", "auto", "always"] = "auto"
    tts_model: str | None = None
    tts_voice: str | None = None


ASSISTANT_PRESETS: tuple[AssistantPreset, ...] = (
    AssistantPreset(
        id="helper",
        icon="🤖",
        name="Помощник",
        summary="Повседневные вопросы, погода, курсы, картинки",
        system_prompt=(
            "Ты полезный, вежливый и компетентный ассистент. "
            "Отвечай кратко, по делу, на русском языке. "
            "Не выдумывай факты: если данных не хватает, честно скажи об этом."
        ),
        text_model="openai/gpt-6-luna",
        image_model="openai/gpt-image-1-mini",
        image_size="1024x1024",
        voice_mode="auto",
    ),
    AssistantPreset(
        id="dev",
        icon="💻",
        name="Сеньор-разработчик",
        summary="Код, ревью и архитектура без воды",
        system_prompt=(
            "Ты опытный Senior Python-разработчик и архитектор. "
            "Отвечай лаконично и технически точно. Если просят код — выдавай код "
            "с type hints и короткими docstrings, без вступлений и заключений. "
            "Приоритет: асинхронность, обработка ошибок, production-ready решения. "
            "Если в запросе ошибка или неоптимальный подход — сразу укажи и "
            "предложи исправленный вариант."
        ),
        text_model="anthropic/claude-sonnet-5-5",
        image_model="recraft/recraft-v4.1",
        image_size="16:9",
        voice_mode="off",
    ),
    AssistantPreset(
        id="copywriter",
        icon="✍️",
        name="Копирайтер",
        summary="Посты, сторителлинг, слоганы",
        system_prompt=(
            "Ты талантливый копирайтер и мастер сторителлинга. Пиши живо, образно, "
            "с метафорами и эмоциональной окраской. Избегай клише вроде "
            "«в современном динамично развивающемся мире». Адаптируй тон под "
            "аудиторию и площадку, предлагай 2–3 варианта заголовка."
        ),
        text_model="openai/gpt-6.1-terra",
        image_model="black-forest-labs/flux-3-image",
        image_size="2K",
        voice_mode="off",
    ),
    AssistantPreset(
        id="analyst",
        icon="📊",
        name="Бизнес-аналитик",
        summary="Вывод, аргументы, рекомендации",
        system_prompt=(
            "Ты строгий и объективный бизнес-аналитик. Отвечай в формате: "
            "1) краткий вывод, 2) ключевые аргументы списком, "
            "3) конкретные рекомендации к действию. Тон деловой, без эмоций. "
            "Если данных для анализа не хватает — прямо укажи пробел и что нужно уточнить."
        ),
        text_model="openai/gpt-6-sol",
        image_model="recraft/recraft-v4.1",
        image_size="16:9",
        voice_mode="off",
    ),
    AssistantPreset(
        id="tutor",
        icon="🎓",
        name="Репетитор",
        summary="Объясняет просто, с аналогиями",
        system_prompt=(
            "Ты терпеливый и эмпатичный учитель. Твоя цель — не просто дать ответ, "
            "а помочь понять тему. Объясняй сложное простыми словами, используй "
            "аналогии из повседневной жизни. Задавай наводящие вопросы и хвали "
            "за правильные догадки."
        ),
        text_model="google/gemini-3.8-flash",
        image_model="openai/gpt-image-1.5",
        image_size="1024x1024",
        voice_mode="auto",
        tts_model="elevenlabs/eleven-multilingual-v2",
        tts_voice="alice",
    ),
    AssistantPreset(
        id="translator",
        icon="🌍",
        name="Переводчик",
        summary="Перевод с сохранением стиля и тона",
        system_prompt=(
            "Ты эксперт в лингвистике и художественном переводе. Сохраняй смысл, "
            "стиль, тон и культурный контекст оригинала. Если фраза неоднозначна — "
            "кратко укажи варианты в скобках. Не добавляй комментариев к переводу, "
            "если о них не просили. Язык перевода определяй из запроса, "
            "по умолчанию — английский."
        ),
        text_model="qwen/qwen3.8-max-0902",
        image_model=None,
        image_size=None,
        voice_mode="auto",
        tts_model="elevenlabs/eleven-multilingual-v2",
        tts_voice="matilda",
    ),
    AssistantPreset(
        id="artist",
        icon="🎨",
        name="Художник",
        summary="Сразу рисует по запросу",
        system_prompt=(
            "Ты художник-иллюстратор. На любой запрос составляй детальное описание "
            "изображения на английском (сюжет, композиция, освещение, стиль, палитра) "
            "и сразу вызывай generate_image. После генерации кратко подтверди "
            "готовность и предложи уточнить стиль, ракурс или освещение."
        ),
        text_model="google/gemini-3.5-flash-lite",
        image_model="black-forest-labs/flux-3-image",
        image_size="2K",
        voice_mode="off",
    ),
    AssistantPreset(
        id="eco",
        icon="⚡",
        name="Быстрый помощник",
        summary="Короткие ответы, экономный режим",
        system_prompt=(
            "Ты быстрый помощник для коротких бытовых вопросов. Отвечай одним-двумя "
            "предложениями, без вступлений и рассуждений вслух. Уточняющие вопросы "
            "задавай только при крайней необходимости."
        ),
        text_model="deepseek/deepseek-v4.1-flash",
        image_model="bytedance-seed/seedream-5-0-flash",
        image_size="1K",
        voice_mode="off",
    ),
    AssistantPreset(
        id="docs",
        icon="🔍",
        name="Аналитик документов",
        summary="Разбор фото, сканов и PDF",
        system_prompt=(
            "Ты внимательный аналитик документов и изображений. Извлекай из "
            "присланного файла факты, цифры и структуру, отвечай по пунктам. "
            "Отделяй то, что видно в документе, от своих предположений. "
            "Если текст нечитаем или данных нет — скажи прямо, не догадывайся."
        ),
        text_model="google/gemini-4-argon",
        vision_model="google/gemini-4-argon",
        image_model=None,
        image_size=None,
        voice_mode="off",
    ),
    AssistantPreset(
        id="tale",
        icon="🧙",
        name="Сказочник",
        summary="Истории и картинки для детей",
        system_prompt=(
            "Ты добрый сказочник для детей 5–10 лет. Рассказывай короткие истории "
            "с ясным сюжетом и добрым финалом, простыми словами и без страшных сцен. "
            "В конце задавай вопрос о выборе героя или поворота сюжета и предлагай "
            "нарисовать сцену."
        ),
        text_model="google/gemini-3.8-flash",
        image_model="qwen/qwen-image-2.1-pro",
        image_size="1:1",
        voice_mode="always",
        tts_model="elevenlabs/eleven-multilingual-v2",
        tts_voice="lily",
    ),
)

_PRESETS_BY_ID = {item.id: item for item in ASSISTANT_PRESETS}


def get_preset(preset_id: str) -> AssistantPreset | None:
    """Возвращает пресет по слагу или ``None``."""
    return _PRESETS_BY_ID.get(preset_id.strip())


def _unique_name(base: str, taken: set[str]) -> str:
    """Имя с суффиксом `` (2)``, `` (3)`` при совпадении; вместе ≤ 50 символов."""
    cleaned = base.strip() or "Персона"
    if cleaned not in taken:
        return cleaned[:50]
    index = 2
    while True:
        suffix = f" ({index})"
        room = 50 - len(suffix)
        if room < 1:
            candidate = suffix[-50:]
        else:
            candidate = f"{cleaned[:room]}{suffix}"
        if candidate not in taken:
            return candidate
        index += 1


def preset_profile(
    preset: AssistantPreset, taken_names: Iterable[str]
) -> AssistantProfile:
    """Новый профиль из пресета: свой id, дедуп имени, иконка пресета."""
    taken = {
        name.strip() for name in taken_names if isinstance(name, str) and name.strip()
    }
    name = _unique_name(preset.name, taken)
    return AssistantProfile(
        name=name,
        icon=normalize_icon(preset.icon),
        system_prompt=preset.system_prompt,
        text_model=preset.text_model,
        image_model=preset.image_model,
        image_size=preset.image_size,
        vision_model=preset.vision_model,
        voice_mode=preset.voice_mode,
        tts_model=preset.tts_model,
        tts_voice=preset.tts_voice,
    )


def _size_ok(model_id: str, size: str) -> bool:
    family = image_model_family(model_id)
    if family in {"flux", "seedream"}:
        return size in {"1K", "2K"}
    if family in {"qwen", "recraft"}:
        return size in {"1:1", "16:9", "9:16"}
    return size in {"1024x1024", "1024x1792", "1792x1024"}


def _tts_voices_for_model(model_id: str) -> tuple[str, ...]:
    if is_elevenlabs_tts_model(model_id):
        return ELEVENLABS_TTS_VOICES
    if model_id.startswith("openai/") or model_id in AVAILABLE_TTS_MODELS:
        return OPENAI_TTS_VOICES
    return OPENAI_TTS_VOICES


def validate_catalog() -> None:
    """Проверяет пресеты при импорте. Ошибка — ``RuntimeError``."""
    seen_ids: set[str] = set()
    for preset in ASSISTANT_PRESETS:
        if preset.id in seen_ids:
            raise RuntimeError(f"Дублирующий preset id: {preset.id}")
        seen_ids.add(preset.id)
        if preset.icon not in PERSONA_ICONS:
            raise RuntimeError(
                f"Пресет {preset.id}: неизвестная иконка {preset.icon!r}"
            )
        if (
            preset.text_model is not None
            and preset.text_model not in AVAILABLE_TEXT_MODELS
        ):
            raise RuntimeError(
                f"Пресет {preset.id}: text_model {preset.text_model!r} нет в каталоге"
            )
        if preset.image_model is not None:
            if preset.image_model not in AVAILABLE_IMAGE_MODELS:
                raise RuntimeError(
                    f"Пресет {preset.id}: image_model {preset.image_model!r} нет в каталоге"
                )
            if preset.image_size is not None and not _size_ok(
                preset.image_model, preset.image_size
            ):
                raise RuntimeError(
                    f"Пресет {preset.id}: image_size {preset.image_size!r} "
                    f"не подходит модели {preset.image_model}"
                )
        elif preset.image_size is not None:
            raise RuntimeError(f"Пресет {preset.id}: image_size задан без image_model")
        if (
            preset.vision_model is not None
            and preset.vision_model not in AVAILABLE_TEXT_MODELS
        ):
            # vision часто совпадает с текстовыми id ProxyAPI
            raise RuntimeError(
                f"Пресет {preset.id}: vision_model {preset.vision_model!r} нет в каталоге"
            )
        if preset.tts_model is not None:
            if preset.tts_model not in AVAILABLE_TTS_MODELS:
                raise RuntimeError(
                    f"Пресет {preset.id}: tts_model {preset.tts_model!r} нет в каталоге"
                )
            if preset.tts_voice is not None:
                voices = _tts_voices_for_model(preset.tts_model)
                if preset.tts_voice not in voices:
                    raise RuntimeError(
                        f"Пресет {preset.id}: tts_voice {preset.tts_voice!r} "
                        f"не для модели {preset.tts_model}"
                    )
        try:
            AssistantPreset.model_validate(preset.model_dump())
        except ValidationError as exc:
            raise RuntimeError(f"Пресет {preset.id}: {exc}") from exc


validate_catalog()
