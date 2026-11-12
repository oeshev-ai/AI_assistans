"""Compatibility shim — implementation lives in ``app.bot``."""

from __future__ import annotations

import asyncio
import sys

from app.bot.constants import TELEGRAM_TEXT_LIMIT  # noqa: F401
from app.bot.main import main
from app.bot.models import (  # noqa: F401
    _fit_screen,
    _hints_block,
    _image_model_of,
    _should_send_voice_reply,
    _text_model_of,
)
from app.bot.runtime import (  # noqa: F401
    _compact_tool_content,
    _extract_generated_images,
    _image_completion_text,
    _image_prompts_from_messages,
    _remember_model_turn,
    _sanitize_history_message,
)

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
