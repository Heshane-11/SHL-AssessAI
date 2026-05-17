"""
Utility helpers: JSON parsing, text normalization, logging.
"""

import json
import re
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Robustly extract a JSON object from an LLM response.
    Handles markdown code fences and stray text around the JSON.
    """
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.strip().strip("`").strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the outermost { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    logger.warning("Failed to extract JSON from LLM response: %s", text[:200])
    return None


def safe_response(
    reply: str,
    recommendations: Optional[list] = None,
    end_of_conversation: bool = False,
) -> Dict[str, Any]:
    """Build a guaranteed-valid response dict."""
    return {
        "reply": reply,
        "recommendations": recommendations or [],
        "end_of_conversation": end_of_conversation,
    }


def truncate_text(text: str, max_chars: int = 400) -> str:
    """Truncate text for use in prompts to save tokens."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def normalize_test_type(raw: str) -> str:
    """Normalize test type codes to single uppercase letter."""
    mapping = {
        "ability": "A",
        "aptitude": "A",
        "biodata": "B",
        "situational": "B",
        "competenc": "C",
        "development": "D",
        "360": "D",
        "exercise": "E",
        "knowledge": "K",
        "skill": "K",
        "motivation": "M",
        "personality": "P",
        "behavior": "P",
        "behaviour": "P",
        "simulation": "S",
    }
    raw_lower = raw.lower().strip()
    for key, code in mapping.items():
        if key in raw_lower:
            return code
    # If raw is already a single letter code, return as uppercase
    if len(raw_lower) == 1 and raw_lower.upper() in "ABCDEKMP S":
        return raw_lower.upper()
    return raw.strip() or "K"


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
    )
