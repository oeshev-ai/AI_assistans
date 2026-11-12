"""FSM state groups."""

from __future__ import annotations

from aiogram.fsm.state import State, StatesGroup


class NewAssistantSG(StatesGroup):
    """Шаги создания новой персоны."""

    name = State()
    prompt = State()
    models = State()


class EditAssistantSG(StatesGroup):
    """Правка имени или системного промпта существующей персоны."""

    name = State()
    prompt = State()


class NewTopicSG(StatesGroup):
    """Шаг ввода названия новой темы."""

    name = State()


class RenameTopicSG(StatesGroup):
    """Шаг ввода нового названия темы."""

    name = State()


class ImageSG(StatesGroup):
    """Экран выбора модели и размера, затем описание картинки."""

    config = State()
    prompt = State()
