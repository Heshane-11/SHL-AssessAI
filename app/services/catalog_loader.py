"""
Catalog loader — reads the JSON catalog from disk and exposes helpers.
"""

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set

from app.config import get_settings
from app.models.catalog import CatalogItem

logger = logging.getLogger(__name__)
settings = get_settings()


@lru_cache(maxsize=1)
def load_catalog() -> List[CatalogItem]:
    """Load and cache the SHL catalog from disk."""
    path = settings.catalog_file
    if not path.exists():
        logger.error("Catalog file not found at %s. Run scripts/scrape_catalog.py first.", path)
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw: List[Dict] = json.load(f)
    items = [CatalogItem(**item) for item in raw]
    logger.info("Loaded %d assessments from catalog.", len(items))
    return items


def get_catalog_names() -> Set[str]:
    """Return all assessment names for hallucination checks."""
    return {item.name for item in load_catalog()}


def get_item_by_name(name: str) -> Optional[CatalogItem]:
    """Case-insensitive lookup of a catalog item by name."""
    name_lower = name.lower().strip()
    for item in load_catalog():
        if item.name.lower().strip() == name_lower:
            return item
    return None


def get_items_by_names(names: List[str]) -> List[CatalogItem]:
    """Bulk lookup."""
    return [item for name in names if (item := get_item_by_name(name))]


def format_catalog_context(items: List[CatalogItem], max_items: int = 15) -> str:
    """
    Format catalog items as a compact text block for injection into LLM prompts.
    """
    lines = []
    for i, item in enumerate(items[:max_items], 1):
        skills = ", ".join(item.skills_measured[:5]) if item.skills_measured else "N/A"
        lines.append(
            f"{i}. [{item.test_type}] {item.name}\n"
            f"   URL: {item.url}\n"
            f"   Duration: {item.duration or 'N/A'} | Remote: {item.remote_testing or 'N/A'}\n"
            f"   Skills: {skills}\n"
            f"   Description: {item.description[:150]}..."
        )
    return "\n".join(lines)
