"""
Clarification engine — asks targeted questions when user query is vague.
"""

import logging
from typing import Dict, Any, List

from app.models.schemas import Message
from app.prompts.templates import CLARIFICATION_PROMPT_TEMPLATE
from app.services.intent_detector import IntentResult
from app.services.llm_client import call_llm_json
from app.utils.helpers import safe_response

logger = logging.getLogger(__name__)


def _identify_missing_info(intent: IntentResult) -> str:
    """Identify what information is missing for a recommendation."""
    missing: List[str] = []
    if not intent.role:
        missing.append("job role")
    if not intent.seniority:
        missing.append("seniority level (junior/mid/senior)")
    if not intent.skills:
        missing.append("key skills or competencies required")
    return ", ".join(missing) if missing else "specific requirements"


def clarify(
    messages: List[Message],
    intent: IntentResult,
) -> Dict[str, Any]:
    """
    Generate a clarifying question when the user's request lacks detail.
    Never asks more than 2 questions per response.
    """
    latest_user_msg = next(
        (m.content for m in reversed(messages) if m.role == "user"), ""
    )

    # Count how many clarifications we've already asked
    assistant_questions = sum(
        1
        for m in messages
        if m.role == "assistant" and "?" in m.content
    )

    # If we've already asked 2 rounds and still vague, make a best-effort recommendation
    if assistant_questions >= 2:
        logger.info("Max clarifications reached, proceeding with best-effort recommendation.")
        return safe_response(
            "Based on what you've shared, let me suggest some general assessments. "
            "You can refine these by giving me more details about the role."
        )

    missing_info = _identify_missing_info(intent)

    prompt = CLARIFICATION_PROMPT_TEMPLATE.format(
        user_message=latest_user_msg,
        missing_info=missing_info,
    )

    result = call_llm_json(prompt)
    result["recommendations"] = []  # Always empty during clarification
    return result
