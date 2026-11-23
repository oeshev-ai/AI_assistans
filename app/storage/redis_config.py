"""Настройки Redis для профилей ассистентов.

Пустой ``redis_url`` означает память процесса. Темы читают свой
``TopicConfig``, в том числе ``TOPIC_TTL_DAYS`` и ``MAX_HISTORY_LENGTH``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from app.infra.settings import get_settings, load_project_dotenv


class RedisConfig(BaseModel):
    """URL Redis. Пустой URL — память процесса."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    redis_url: str = ""

    @classmethod
    def from_env(cls) -> RedisConfig:
        """Читает ``REDIS_URL`` из общего слоя настроек."""
        load_project_dotenv()
        return cls(redis_url=get_settings().redis_url)
