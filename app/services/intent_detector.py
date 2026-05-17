"""
Intent detection — classifies user messages to guide conversation flow.
Uses Gemini for semantic understanding, with a fast regex pre-filter.
"""

import json
import logging
import re
from typing import List, Optional

from app.models.schemas import Message
from app.prompts.templates import INTENT_DETECTION_PROMPT
from app.services.llm_client import call_llm
from app.utils.helpers import extract_json_from_text

logger = logging.getLogger(__name__)

# ── Quick regex shortcuts (avoids LLM call for obvious cases) ───────────────
_COMPARE_RE = re.compile(
    r"\b(compare|difference between|vs\.?|versus|which is better|"
    r"what('s| is) the difference)\b",
    re.IGNORECASE,
)
_INJECTION_QUICK_RE = re.compile(
    r"\b(ignore|forget|disregard|override)\b.*\b(instruction|prompt|rule|system)\b",
    re.IGNORECASE,
)
_VAGUE_RE = re.compile(
    r"^(hi|hello|hey|i need (an?|some)? assessment|help me|"
    r"what assessments|can you help)\.?$",
    re.IGNORECASE,
)


class IntentResult:
    def __init__(
        self,
        intent: str,
        role: Optional[str] = None,
        skills: Optional[List[str]] = None,
        seniority: Optional[str] = None,
        is_vague: bool = False,
    ):
        self.intent = intent
        self.role = role
        self.skills = skills or []
        self.seniority = seniority
        self.is_vague = is_vague

    def __repr__(self) -> str:
        return (
            f"IntentResult(intent={self.intent!r}, role={self.role!r}, "
            f"skills={self.skills}, seniority={self.seniority!r}, is_vague={self.is_vague})"
        )


def detect_intent(messages: List[Message]) -> IntentResult:
    """
    Detect user intent from the conversation.
    Uses fast regex first, then LLM for nuanced cases.
    """
    latest = messages[-1].content if messages else ""

    # ── Fast path: injection ────────────────────────────────────────────────
    if _INJECTION_QUICK_RE.search(latest):
        return IntentResult(intent="injection")

    # ── Fast path: compare ──────────────────────────────────────────────────
    if _COMPARE_RE.search(latest):
        return IntentResult(intent="compare")

    # ── Fast path: vague one-liners ─────────────────────────────────────────
    if _VAGUE_RE.match(latest.strip()):
        return IntentResult(intent="clarify_needed", is_vague=True)

    # ── LLM-based intent detection ──────────────────────────────────────────
    snippet = _build_snippet(messages)
    prompt = INTENT_DETECTION_PROMPT.format(
        conversation_snippet=snippet,
        latest_message=latest,
    )

    raw = call_llm(prompt)
    if not raw:
        # Default to recommend if LLM is unavailable
        return IntentResult(intent="recommend", is_vague=True)

    parsed = extract_json_from_text(raw)
    if not parsed:
        return IntentResult(intent="recommend", is_vague=True)

    return IntentResult(
        intent=parsed.get("intent", "recommend"),
        role=parsed.get("role"),
        skills=parsed.get("skills", []),
        seniority=parsed.get("seniority"),
        is_vague=parsed.get("is_vague", False),
    )


def _build_snippet(messages: List[Message], last_n: int = 6) -> str:
    """Build a short conversation snippet for intent prompt."""
    recent = messages[-last_n:]
    lines = []
    for msg in recent:
        role = "User" if msg.role == "user" else "Assistant"
        lines.append(f"{role}: {msg.content[:300]}")
    return "\n".join(lines)


def extract_comparison_targets(text: str) -> List[str]:
    """
    Extract assessment names to compare from user text.
    E.g. "Compare OPQ and GSA" → ["OPQ", "GSA"]
    """
    # Pattern: compare X and Y / difference between X and Y
    pattern = re.compile(
        r"(?:compare|between|vs\.?|versus)\s+(.+?)(?:\s+and\s+|\s+vs\.?\s+)(.+?)(?:\?|$)",
        re.IGNORECASE,
    )
    match = pattern.search(text)
    if match:
        return [match.group(1).strip(), match.group(2).strip()]

    # Fallback: look for quoted or title-case words
    quoted = re.findall(r'"([^"]+)"', text)
    if len(quoted) >= 2:
        return quoted[:2]

    return []
