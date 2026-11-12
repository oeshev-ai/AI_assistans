"""Input validation and prompt-injection sanitization."""

from __future__ import annotations

import re

import structlog
from pydantic import BaseModel, ConfigDict, Field, ValidationError

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


class UserTextInput(BaseModel):
    """Validated free-form chat text."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    text: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)


class ImagePromptInput(BaseModel):
    """Validated image-generation prompt."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_LENGTH)


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

    @staticmethod
    def contains_sensitive_data(text: str) -> bool:
        """Heuristic check for card-like digit groups."""
        sensitive_patterns = [
            r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
            r"\b\d{16}\b",
        ]
        return any(re.search(pattern, text) for pattern in sensitive_patterns)

    @staticmethod
    def validate_user_text(text: str) -> str:
        """Pydantic-validate chat text and return the cleaned value."""
        try:
            return UserTextInput(text=text).text
        except ValidationError as exc:
            raise ValueError("Сообщение слишком длинное или пустое.") from exc

    @staticmethod
    def validate_image_prompt(prompt: str) -> str:
        """Pydantic-validate an image prompt and return the cleaned value."""
        try:
            return ImagePromptInput(prompt=prompt).prompt
        except ValidationError as exc:
            raise ValueError("Промпт слишком длинный или пустой.") from exc


security = SecurityValidator()
