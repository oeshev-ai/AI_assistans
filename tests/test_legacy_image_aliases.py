"""Снятые image-id читаются как актуальные, неизвестные — нет."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.bot.models import _image_model_of
from app.llm.assist_ai import AssistAIConfig
from app.storage.assistant_storage import (
    DEFAULT_PERSONA_PROMPT,
    LEGACY_IMAGE_MODEL_ALIASES,
    AssistantProfile,
)


def _assistant(*, aliases: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            default_image_model="openai/gpt-image-2.6-flare",
            legacy_model_aliases=aliases,
        )
    )


@pytest.mark.parametrize("old", sorted(LEGACY_IMAGE_MODEL_ALIASES))
def test_deprecated_ids_alias_to_current(old: str) -> None:
    profile = AssistantProfile(
        name="Старый",
        system_prompt=DEFAULT_PERSONA_PROMPT,
        image_model=old,
    )
    assert profile.image_model == LEGACY_IMAGE_MODEL_ALIASES[old]


def test_unknown_image_id_stays_and_request_uses_default() -> None:
    profile = AssistantProfile(
        name="Старый",
        system_prompt=DEFAULT_PERSONA_PROMPT,
        image_model="vendor/removed-image-model",
    )
    assert profile.image_model == "vendor/removed-image-model"
    assert (
        _image_model_of(profile, _assistant(), chat_id=1)  # type: ignore[arg-type]
        == "openai/gpt-image-2.6-flare"
    )


def test_alias_flag_off_keeps_old_id_until_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IMAGE_LEGACY_ALIASES", "0")
    profile = AssistantProfile(
        name="Старый",
        system_prompt=DEFAULT_PERSONA_PROMPT,
        image_model="openai/gpt-image-1.5",
    )
    assert profile.image_model == "openai/gpt-image-1.5"
    resolved = _image_model_of(profile, _assistant(aliases=False), chat_id=3)  # type: ignore[arg-type]
    assert resolved == "openai/gpt-image-2.6-flare"


def test_from_env_without_image_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.llm.assist_ai.load_dotenv", lambda *_args, **_kwargs: False
    )
    for name in (
        "PROXYAPI_IMAGE_MODEL",
        "PROXYAPI_IMAGE_EDIT_MODEL",
        "IMAGE_COMPLEXITY_EDIT",
        "IMAGE_QUALITY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PROXYAPI_API_KEY", "test-key")
    config = AssistAIConfig.from_env()
    assert config.default_image_model == "openai/gpt-image-2.6-flare"
    assert config.default_image_edit_model == "openai/gpt-image-2.6-flare"
    assert config.image_complexity_edit == "low"
