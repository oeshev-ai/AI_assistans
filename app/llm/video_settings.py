"""Настройки ролика персоны: формат, длина, разрешение и звук Seedance 2.

Каталог совпадает с гайдом Polza для ``bytedance/seedance-2``:
форматы ``16:9`` … ``21:9``, длина 4–15 с, разрешение ``480p`` или ``720p``,
``generate_audio`` true/false. Пустые поля профиля означают эти дефолты.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass

VIDEO_ASPECT_RATIOS: tuple[str, ...] = ("16:9", "9:16", "1:1", "4:3", "3:4", "21:9")
VIDEO_DURATIONS: tuple[int, ...] = (4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
VIDEO_RESOLUTIONS: tuple[str, ...] = ("480p", "720p")

DEFAULT_VIDEO_ASPECT_RATIO = "16:9"
DEFAULT_VIDEO_DURATION = 5
DEFAULT_VIDEO_RESOLUTION = "720p"
DEFAULT_VIDEO_GENERATE_AUDIO = False


@dataclass(frozen=True)
class VideoSettings:
    """Значения, которые уходят в запрос генерации ролика."""

    aspect_ratio: str
    duration: int
    resolution: str
    generate_audio: bool


def coerce_aspect_ratio(value: object) -> str | None:
    """Допустимый формат или ``None``, если значение пустое либо чужое."""
    if value is None:
        return None
    text = str(value).strip()
    if text in VIDEO_ASPECT_RATIOS:
        return text
    return None


def coerce_duration(value: object) -> int | None:
    """Секунды из каталога. ``bool`` не считается числом."""
    if value is None or isinstance(value, bool):
        return None
    number: int | None = None
    if isinstance(value, int):
        number = value
    elif isinstance(value, float) and value.is_integer():
        number = int(value)
    elif isinstance(value, str) and value.strip().isdigit():
        number = int(value.strip())
    if number in VIDEO_DURATIONS:
        return number
    return None


def coerce_resolution(value: object) -> str | None:
    """``480p`` или ``720p``. Регистр не важен."""
    if value is None:
        return None
    text = str(value).strip().casefold()
    if text in VIDEO_RESOLUTIONS:
        return text
    return None


def coerce_generate_audio(value: object) -> bool | None:
    """Булево звука. Незнакомая строка отбрасывается, а не становится False."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().casefold()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return None


def resolve_video_settings(
    *,
    aspect_ratio: str | None = None,
    duration: int | None = None,
    resolution: str | None = None,
    generate_audio: bool | None = None,
) -> VideoSettings:
    """Подставляет дефолты Seedance на место пустых полей."""
    return VideoSettings(
        aspect_ratio=coerce_aspect_ratio(aspect_ratio) or DEFAULT_VIDEO_ASPECT_RATIO,
        duration=coerce_duration(duration) or DEFAULT_VIDEO_DURATION,
        resolution=coerce_resolution(resolution) or DEFAULT_VIDEO_RESOLUTION,
        generate_audio=(
            DEFAULT_VIDEO_GENERATE_AUDIO
            if generate_audio is None
            else bool(generate_audio)
        ),
    )


def video_settings_from(profile: object | None) -> VideoSettings:
    """Настройки ролика профиля. Без профиля — дефолты."""
    if profile is None:
        return resolve_video_settings()
    return resolve_video_settings(
        aspect_ratio=getattr(profile, "video_aspect_ratio", None),
        duration=getattr(profile, "video_duration", None),
        resolution=getattr(profile, "video_resolution", None),
        generate_audio=getattr(profile, "video_generate_audio", None),
    )


def video_settings_summary(settings: VideoSettings) -> str:
    """Короткая строка для экрана персоны."""
    sound = "со звуком" if settings.generate_audio else "без звука"
    return (
        f"{settings.aspect_ratio}, {settings.duration} с, "
        f"{settings.resolution}, {sound}"
    )


def parse_video_callback(data: str) -> tuple[str, str | int | bool, str] | None:
    """``video:{kind}:{value}`` → поле профиля, значение и текст всплывающего ответа.

    Неизвестный фрагмент возвращает ``None``, чтобы кнопка не записала мусор.
    """
    raw = data.removeprefix("video:")
    kind, _, value = raw.partition(":")
    if not value:
        return None
    if kind == "aspect":
        aspect = coerce_aspect_ratio(value)
        if aspect is None:
            return None
        return "video_aspect_ratio", aspect, f"Формат: {aspect}"
    if kind == "dur":
        duration = coerce_duration(value)
        if duration is None:
            return None
        return "video_duration", duration, f"Длительность: {duration} с"
    if kind == "res":
        resolution = coerce_resolution(value)
        if resolution is None:
            return None
        return "video_resolution", resolution, f"Разрешение: {resolution}"
    if kind == "audio":
        audio = coerce_generate_audio(value)
        if audio is None or value not in {"0", "1"}:
            return None
        label = "вкл" if audio else "выкл"
        return "video_generate_audio", audio, f"Звук: {label}"
    return None


_REQUEST_VIDEO: ContextVar[VideoSettings | None] = ContextVar(
    "request_video_settings", default=None
)


class VideoSettingsScope:
    """На время хода подставляет настройки ролика активной персоны."""

    def __init__(self, settings: VideoSettings) -> None:
        self._settings = settings
        self._token: Token[VideoSettings | None] | None = None

    def __enter__(self) -> VideoSettingsScope:
        self._token = _REQUEST_VIDEO.set(self._settings)
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._token is not None:
            _REQUEST_VIDEO.reset(self._token)


def current_video_settings() -> VideoSettings | None:
    """Настройки хода. Вне ``VideoSettingsScope`` — ``None``."""
    return _REQUEST_VIDEO.get()
