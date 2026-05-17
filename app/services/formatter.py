"""
Response formatter — guarantees the output always matches the ChatResponse schema.
"""

import logging
from typing import Any, Dict, List

from app.models.schemas import Assessment, ChatResponse

logger = logging.getLogger(__name__)


def format_response(raw: Dict[str, Any]) -> ChatResponse:
    """
    Convert a raw LLM result dict into a validated ChatResponse.
    Always succeeds — falls back to safe defaults on bad input.
    """
    reply = raw.get("reply", "")
    if not isinstance(reply, str) or not reply.strip():
        reply = "I encountered an issue generating a response. Please try again."

    raw_recs = raw.get("recommendations", [])
    if not isinstance(raw_recs, list):
        raw_recs = []

    assessments: List[Assessment] = []
    for rec in raw_recs:
        if not isinstance(rec, dict):
            continue
        name = rec.get("name", "").strip()
        url = rec.get("url", "").strip()
        test_type = rec.get("test_type", "K").strip()

        if not name or not url:
            continue

        # Validate URL starts with https://
        if not url.startswith("https://"):
            logger.warning("Skipping assessment with invalid URL: %s", url)
            continue

        assessments.append(
            Assessment(name=name, url=url, test_type=test_type or "K")
        )

    end_of_conversation = bool(raw.get("end_of_conversation", False))

    # Enforce schema rules
    if len(assessments) > 10:
        assessments = assessments[:10]

    return ChatResponse(
        reply=reply.strip(),
        recommendations=assessments,
        end_of_conversation=end_of_conversation,
    )
