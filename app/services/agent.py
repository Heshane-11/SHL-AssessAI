"""
The main chat orchestrator — routes intents to the correct service.
"""

import logging
from typing import Dict, Any

from app.models.schemas import ChatRequest, ChatResponse
from app.guards.safety import run_all_guards
from app.services.intent_detector import detect_intent
from app.services.recommender import recommend, refine
from app.services.comparator import compare
from app.services.clarifier import clarify
from app.services.formatter import format_response
from app.prompts.templates import REFUSAL_PROMPT_TEMPLATE
from app.services.llm_client import call_llm_json
from app.utils.helpers import safe_response

logger = logging.getLogger(__name__)


def handle_chat(request: ChatRequest) -> ChatResponse:
    """
    Main conversation handler.

    Pipeline:
    1. Run safety guards on latest user message
    2. Detect intent from conversation history
    3. Route to: clarify / recommend / refine / compare / refuse
    4. Format and validate the response
    """
    messages = request.messages
    latest_user_msg = next(
        (m.content for m in reversed(messages) if m.role == "user"), ""
    )

    # ── Step 1: Safety guards ────────────────────────────────────────────────
    blocked, reason = run_all_guards(latest_user_msg)
    if blocked:
        refusal_type = (
            "prompt injection attempt" if reason == "prompt_injection" else "out-of-scope request"
        )
        prompt = REFUSAL_PROMPT_TEMPLATE.format(request_type=refusal_type)
        raw = call_llm_json(prompt)
        raw["recommendations"] = []
        return format_response(raw)

    # ── Step 2: Intent detection ─────────────────────────────────────────────
    intent = detect_intent(messages)
    logger.info("Detected intent: %s", intent)

    # ── Step 3: Route to handler ─────────────────────────────────────────────
    raw: Dict[str, Any]

    if intent.intent == "injection":
        raw = safe_response(
            "I'm only able to help with SHL assessment recommendations. "
            "I can't follow instructions that attempt to change my behavior."
        )

    elif intent.intent == "out_of_scope":
        prompt = REFUSAL_PROMPT_TEMPLATE.format(request_type="non-SHL request")
        raw = call_llm_json(prompt)
        raw["recommendations"] = []

    elif intent.intent == "clarify_needed":
        raw = clarify(messages, intent)

    elif intent.intent == "compare":
        raw = compare(messages)

    elif intent.intent == "refine":
        raw = refine(messages, intent)

    else:  # "recommend" (default)
        raw = recommend(messages, intent)

    # ── Step 4: Format + validate ────────────────────────────────────────────
    return format_response(raw)
