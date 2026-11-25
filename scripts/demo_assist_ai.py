"""Ручная проверка AssistAI: погода, валюта и одна картинка.

Реальные платные вызовы. Запуск из корня репозитория:

    python scripts/demo_assist_ai.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import structlog
from pydantic import ValidationError

from app.infra.settings import load_project_dotenv
from app.llm.assist_ai import AssistAIClient
from app.llm.errors import AssistAIError

logger = structlog.get_logger("AssistAI")


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def _describe_image(image: str) -> str:
    if image.startswith(("http://", "https://")):
        return image
    return f"base64, {len(image)} символов"


async def main() -> None:
    """Проверяет Function Calling на погоде и валюте и один запрос картинки."""
    load_project_dotenv()
    _configure_logging()
    try:
        client = AssistAIClient()
    except ValidationError as exc:
        logger.error("Некорректная конфигурация AssistAI: %s", exc)
        raise SystemExit(1) from exc

    async with client:
        print("\n--- Погода (Function Calling) ---")
        weather_reply = await client.chat_completion(
            [
                {
                    "role": "user",
                    "content": (
                        "Узнай текущую погоду в Москве через доступную функцию "
                        "и ответь кратко: температура, описание, влажность и ветер."
                    ),
                }
            ]
        )
        print(weather_reply)

        print("\n--- Конвертация валют (Function Calling) ---")
        exchange_reply = await client.chat_completion(
            [
                {
                    "role": "user",
                    "content": (
                        "Конвертируй 100 USD в RUB через доступную функцию. "
                        "Назови курс и итоговую сумму."
                    ),
                }
            ]
        )
        print(exchange_reply)

        print("\n--- Генерация изображения ---")
        images = await client.generate_image(
            "Минималистичный красный маяк на скале, вечерний свет, цифровая иллюстрация"
        )
        for index, image in enumerate(images, start=1):
            print(f"{index}: {_describe_image(image)}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        asyncio.run(main())
    except AssistAIError as exc:
        logger.error("AssistAI: %s", exc.message)
        raise SystemExit(1) from exc
