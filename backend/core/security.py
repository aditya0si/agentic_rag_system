"""
Security utilities — input validation, sanitization, and prompt injection guards.

Protects the RAG pipeline from malicious inputs and prompt injection attacks.
"""

import html
import re
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Maximum allowed lengths
MAX_QUESTION_LENGTH = 2000
MAX_FILENAME_LENGTH = 255
MAX_CHAT_HISTORY_TURNS = 50

# Allowed file extensions (lowercase, with dot)
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

# Patterns that may indicate prompt injection attempts
SUSPICIOUS_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"disregard\s+(all\s+)?prior\s+(prompts|instructions|context)",
    r"you\s+are\s+now\s+(a|an)\s+\w+",  # role hijack
    r"forget\s+(everything|all\s+rules)",
    r"system\s*:\s*",  # fake system message
    r"<\s*system\s*>",
    r"<\s*/?\s*(system|im_start|im_end)\s*>",
    r"reveal\s+(your|the)\s+(system\s+)?prompt",
    r"output\s+your\s+instructions",
]

_compiled_patterns = [re.compile(p, re.IGNORECASE) for p in SUSPICIOUS_PATTERNS]


def sanitize_text(text: str, max_length: int = MAX_QUESTION_LENGTH) -> str:
    """
    Sanitize user-provided text:
      - Strip control characters
      - HTML-escape to prevent XSS in any rendered output
      - Truncate to max_length
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string")

    # Remove null bytes and other control chars (except newline, tab)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Truncate
    if len(text) > max_length:
        logger.warning("input_truncated", original_len=len(text), max_length=max_length)
        text = text[:max_length]

    # HTML-escape to neutralize any embedded markup
    return html.escape(text, quote=False)


def validate_question(question: str) -> str:
    """
    Validate and sanitize a user question.
    Raises ValueError if the question is empty or too long.
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")

    question = question.strip()
    if len(question) > MAX_QUESTION_LENGTH:
        raise ValueError(f"Question exceeds maximum length of {MAX_QUESTION_LENGTH} characters")

    return sanitize_text(question, max_length=MAX_QUESTION_LENGTH)


def validate_filename(filename: str) -> str:
    """
    Validate an uploaded filename.
      - Must have an allowed extension
      - Must not contain path traversal sequences
      - Truncated to MAX_FILENAME_LENGTH
    """
    if not filename or not filename.strip():
        raise ValueError("Filename cannot be empty")

    # Prevent path traversal
    filename = filename.replace("..", "").replace("/", "").replace("\\", "")
    filename = filename.strip()

    if len(filename) > MAX_FILENAME_LENGTH:
        raise ValueError(f"Filename exceeds maximum length of {MAX_FILENAME_LENGTH} characters")

    # Check extension
    lower = filename.lower()
    if not any(lower.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise ValueError(f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}")

    return filename


def detect_prompt_injection(text: str) -> tuple[bool, list[str]]:
    """
    Heuristic detection of prompt injection attempts.

    Returns:
        (is_suspicious, list_of_matched_patterns)
    """
    matched: list[str] = []
    for pattern in _compiled_patterns:
        if pattern.search(text):
            matched.append(pattern.pattern)

    if matched:
        logger.warning(
            "prompt_injection_detected",
            num_matches=len(matched),
            text_preview=text[:100],
        )

    return (len(matched) > 0, matched)


def sanitize_chat_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Sanitize and truncate chat history to prevent context overflow and injection.
    """
    if not history:
        return []

    # Keep only the most recent turns
    if len(history) > MAX_CHAT_HISTORY_TURNS:
        history = history[-MAX_CHAT_HISTORY_TURNS:]
        logger.info("chat_history_truncated", kept_turns=MAX_CHAT_HISTORY_TURNS)

    sanitized: list[dict[str, Any]] = []
    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role not in ("user", "assistant"):
            continue
        sanitized.append(
            {
                "role": role,
                "content": sanitize_text(content, max_length=MAX_QUESTION_LENGTH),
            }
        )

    return sanitized
