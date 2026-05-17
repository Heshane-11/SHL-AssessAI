"""
Recommendation engine — retrieves relevant assessments and formats recommendations.
"""

import logging
from typing import Dict, Any, List

from app.models.schemas import Message
from app.models.catalog import CatalogItem
from app.prompts.templates import RETRIEVAL_PROMPT_TEMPLATE
from app.services.retriever import retriever
from app.services.catalog_loader import format_catalog_context, get_catalog_names
from app.services.llm_client import call_llm_json
from app.services.intent_detector import IntentResult
from app.guards.safety import validate_recommendations
from app.utils.helpers import safe_response, truncate_text

logger = logging.getLogger(__name__)


def build_query_from_intent(intent: IntentResult, messages: List[Message]) -> str:
    """
    Build a rich semantic query from the intent extraction result.
    """
    parts: List[str] = []
    if intent.role:
        parts.append(intent.role)
    if intent.skills:
        parts.extend(intent.skills)
    if intent.seniority:
        parts.append(intent.seniority)

    if parts:
        return " ".join(parts)

    # Fall back to last user message
    user_msgs = [m for m in messages if m.role == "user"]
    return user_msgs[-1].content if user_msgs else "SHL assessment"


def build_conversation_context(messages: List[Message]) -> str:
    """Summarise prior conversation turns for the LLM prompt."""
    lines: List[str] = []
    for msg in messages[-8:]:  # last 8 turns
        role = "User" if msg.role == "user" else "Assistant"
        lines.append(f"{role}: {truncate_text(msg.content, 300)}")
    return "\n".join(lines)


def recommend(
    messages: List[Message],
    intent: IntentResult,
    top_k: int = 10,
) -> Dict[str, Any]:
    """
    Full recommendation pipeline:
    1. Build semantic query
    2. FAISS retrieval
    3. LLM ranking + reply generation
    4. Hallucination guard
    """
    query = build_query_from_intent(intent, messages)
    logger.info("Recommendation query: %s", query)

    # Step 1: Semantic retrieval
    candidates: List[CatalogItem] = retriever.search(query, top_k=top_k)

    if not candidates:
        return safe_response(
            "I couldn't find matching assessments in the catalog. "
            "Could you describe the role or skills in more detail?"
        )

    # Step 2: Build prompt with catalog context
    catalog_context = format_catalog_context(candidates, max_items=top_k)
    conversation_ctx = build_conversation_context(messages)

    prompt = (
        f"Conversation so far:\n{conversation_ctx}\n\n"
        + RETRIEVAL_PROMPT_TEMPLATE.format(
            query=query,
            catalog_context=catalog_context,
        )
    )

    # Step 3: LLM call
    result = call_llm_json(prompt)

    # Step 4: Validate recommendations against catalog
    catalog_names = get_catalog_names()
    raw_recs = result.get("recommendations", [])

    # If LLM returned nothing, use top FAISS results directly
    if not raw_recs:
        raw_recs = [
            {"name": item.name, "url": item.url, "test_type": item.test_type}
            for item in candidates[:top_k]
        ]

    validated = validate_recommendations(raw_recs, catalog_names)

    # Final fallback: top-3 FAISS results if LLM wiped the list
    if not validated:
        validated = [
            {"name": item.name, "url": item.url, "test_type": item.test_type}
            for item in candidates[:3]
        ]

    result["recommendations"] = validated[:10]  # cap at 10
    return result


def refine(
    messages: List[Message],
    intent: IntentResult,
) -> Dict[str, Any]:
    """
    Refinement: treat like a fresh recommendation but with full history for context.
    The LLM will update the shortlist based on additional constraints.
    """
    return recommend(messages, intent, top_k=10)
