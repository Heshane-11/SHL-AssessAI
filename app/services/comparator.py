"""
Comparison engine — generates grounded comparisons between SHL assessments.
"""

import logging
from typing import Dict, Any, List

from app.models.schemas import Message
from app.models.catalog import CatalogItem
from app.prompts.templates import COMPARISON_PROMPT_TEMPLATE
from app.services.catalog_loader import get_item_by_name, load_catalog
from app.services.retriever import retriever
from app.services.llm_client import call_llm_json
from app.services.intent_detector import extract_comparison_targets
from app.utils.helpers import safe_response

logger = logging.getLogger(__name__)


def _format_item_detail(item: CatalogItem) -> str:
    """Format a single catalog item for comparison context."""
    skills = ", ".join(item.skills_measured[:8]) if item.skills_measured else "N/A"
    return (
        f"Name: {item.name}\n"
        f"Type: {item.test_type}\n"
        f"Duration: {item.duration or 'N/A'}\n"
        f"Remote Testing: {item.remote_testing or 'N/A'}\n"
        f"Adaptive/IRT: {item.adaptive_irt or 'N/A'}\n"
        f"Skills Measured: {skills}\n"
        f"Description: {item.description[:300]}"
    )


def _find_assessment(name: str) -> CatalogItem | None:
    """Try exact match first, then fuzzy FAISS search."""
    item = get_item_by_name(name)
    if item:
        return item

    # Fuzzy fallback
    results = retriever.search(name, top_k=1)
    if results:
        logger.info("Fuzzy match for '%s' → '%s'", name, results[0].name)
        return results[0]
    return None


def compare(
    messages: List[Message],
    names: List[str] | None = None,
) -> Dict[str, Any]:
    """
    Build a grounded comparison of two or more SHL assessments.
    """
    latest_user_msg = next(
        (m.content for m in reversed(messages) if m.role == "user"), ""
    )

    # If names not provided, extract from latest message
    targets = names or extract_comparison_targets(latest_user_msg)

    if len(targets) < 2:
        # Try extracting from full history
        full_text = " ".join(m.content for m in messages if m.role == "user")
        targets = extract_comparison_targets(full_text)

    if len(targets) < 2:
        return safe_response(
            "Could you specify which two assessments you'd like to compare? "
            "For example: 'Compare OPQ32 and Verify Numerical Reasoning'."
        )

    # Look up each assessment
    items: List[CatalogItem] = []
    not_found: List[str] = []
    for name in targets[:4]:  # cap at 4 comparisons
        item = _find_assessment(name)
        if item:
            items.append(item)
        else:
            not_found.append(name)

    if len(items) < 2:
        missing = ", ".join(not_found)
        return safe_response(
            f"I couldn't find '{missing}' in the SHL catalog. "
            "Please check the assessment names and try again."
        )

    # Build detailed context
    details = "\n\n---\n\n".join(_format_item_detail(item) for item in items)

    prompt = COMPARISON_PROMPT_TEMPLATE.format(assessment_details=details)
    result = call_llm_json(prompt)

    # Comparisons always return empty recommendations
    result["recommendations"] = []
    return result
