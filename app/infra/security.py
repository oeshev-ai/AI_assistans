"""Input validation and heuristic prompt-injection filtering.

Not a security boundary: length checks matter more; patterns are best-effort.
"""

from __future__ import annotations

import re
import unicodedata

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
    # Common RU paraphrases (heuristic only).
    r"игнорируй\s+(все\s+)?предыдущие\s+инструкции",
    r"забудь\s+(все\s+)?(инструкции|правила)",
    r"системн(ый|ого)\s+промпт",
    r"раскрой\s+(свои\s+)?инструкции",
    r"ты\s+теперь",
]

MAX_MESSAGE_LENGTH = 4000
MAX_PROMPT_LENGTH = 3000

_COMPILED = [re.compile(pattern, re.IGNORECASE) for pattern in MALICIOUS_PATTERNS]
# Mask password in userinfo: redis://:secret@host → redis://:***@host
_URL_PASSWORD = re.compile(r"(://[^/@]*:)([^@]*)(@)")


def redact_url_credentials(url: str) -> str:
    """Mask URL userinfo password for safe logging (never log secrets)."""
    if not url:
        return url
    return _URL_PASSWORD.sub(r"\1***\3", url)


class SecurityValidator:
    """Length checks + heuristic filter before text reaches the LLM."""

    @staticmethod
    def sanitize_prompt(text: str) -> str:
        """Replace known injection phrases. Heuristic — not a hard security gate."""
        original = text
        # NFKC collapses homoglyphs / compatibility forms used in trivial bypasses.
        sanitized = unicodedata.normalize("NFKC", text)
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
