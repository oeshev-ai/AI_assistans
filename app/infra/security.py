"""Input validation and prompt-injection sanitization."""

from __future__ import annotations

import re

import structlog

logger = structlog.get_logger("security")

MALICIOUS_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"forget\s+(everything|all)",
    r"you\s+are\s+now",
    r"new\s+instructions?",
    r"system\s*prompt",
    r"reveal\s+(your|the)\s+instructions",
    r"act\s+as\s+(if|a)",
    r"pretend\s+to\s+be",
    r"disregard\s+(all\s+)?previous",
    r"override\s+(your|the)\s+(instructions|rules)",
]

MAX_MESSAGE_LENGTH = 4000
MAX_PROMPT_LENGTH = 3000

_COMPILED = [re.compile(pattern, re.IGNORECASE) for pattern in MALICIOUS_PATTERNS]


class SecurityValidator:
    """Sanitizes and validates user-provided text before it reaches the LLM."""

    @staticmethod
    def sanitize_prompt(text: str) -> str:
        """Replace known prompt-injection patterns with a filter marker."""
        original = text
        sanitized = text
        matched = 0
        for pattern in _COMPILED:
            updated, count = pattern.subn("[FILTERED]", sanitized)
            if count:
                matched += count
                sanitized = updated
        if matched:
            logger.warning(
                "prompt_injection_attempt_filtered",
                original_length=len(original),
                matches=matched,
            )
        return sanitized

    @staticmethod
    def validate_message_length(text: str) -> bool:
        return len(text) <= MAX_MESSAGE_LENGTH

    @staticmethod
    def validate_prompt_length(prompt: str) -> bool:
        return len(prompt) <= MAX_PROMPT_LENGTH


security = SecurityValidator()
