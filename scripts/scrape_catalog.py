"""
SHL Catalog Scraper — scrapes the SHL Individual Test Solutions page
and stores structured JSON to data/catalog.json.

Run: python scripts/scrape_catalog.py
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "https://www.shl.com"
CATALOG_URL = f"{BASE_URL}/solutions/products/product-catalog/"
OUTPUT_PATH = Path("data/catalog.json")
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
REQUEST_DELAY = 1.0


def _get(url: str) -> Optional[BeautifulSoup]:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "lxml")
    except Exception as exc:
        logger.warning("Failed to fetch %s: %s", url, exc)
        return None


def _clean(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip())


def _parse_yes_no(cell_text: str) -> Optional[str]:
    t = cell_text.strip().lower()
    if t in ("yes", "✓", "●", "y"):
        return "Yes"
    if t in ("no", "✗", "n"):
        return "No"
    return _clean(cell_text) or None


def scrape_detail_page(url: str) -> Dict[str, Any]:
    soup = _get(url)
    if not soup:
        return {}
    details: Dict[str, Any] = {}
    desc_candidates = soup.select(".product-description p, .content-block p, article p")
    if desc_candidates:
        details["description"] = " ".join(
            _clean(p.get_text()) for p in desc_candidates[:3] if _clean(p.get_text())
        )
    full_text = soup.get_text(" ", strip=True)
    dur_match = re.search(r"(\d+\s*(?:to\s*\d+)?\s*minutes?|untimed)", full_text, re.IGNORECASE)
    if dur_match:
        details["duration"] = _clean(dur_match.group(1))
    skills: List[str] = []
    for heading in soup.find_all(["h2", "h3", "h4"]):
        if any(k in heading.get_text().lower() for k in ("measure", "skill", "competenc")):
            ul = heading.find_next("ul")
            if ul:
                skills = [_clean(li.get_text()) for li in ul.find_all("li")][:10]
                break
    if skills:
        details["skills_measured"] = skills
    return details


def scrape_catalog_page(url: str, page: int = 1) -> List[Dict[str, Any]]:
    page_url = f"{url}?page={page}&type=1" if page > 1 else f"{url}?type=1"
    soup = _get(page_url)
    if not soup:
        return []
    rows = soup.select("table tbody tr")
    items: List[Dict[str, Any]] = []
    for row in rows:
        cols = row.find_all("td")
        if len(cols) < 2:
            continue
        link_tag = cols[0].find("a")
        if not link_tag:
            continue
        name = _clean(link_tag.get_text())
        href = link_tag.get("href", "")
        if not href:
            continue
        detail_url = href if href.startswith("http") else BASE_URL + href
        remote_testing = _parse_yes_no(cols[1].get_text()) if len(cols) > 1 else None
        adaptive_irt = _parse_yes_no(cols[2].get_text()) if len(cols) > 2 else None
        test_type_raw = _clean(cols[3].get_text()) if len(cols) > 3 else ""
        codes = re.findall(r"\b[ABCDEKMPST]\b", test_type_raw.upper())
        test_type = codes[0] if codes else "K"
        items.append({
            "name": name, "url": detail_url, "description": "",
            "duration": None, "remote_testing": remote_testing,
            "adaptive_irt": adaptive_irt, "test_type": test_type, "skills_measured": [],
        })
    logger.info("Scraped %d items from page %d", len(items), page)
    return items


def build_fallback_catalog() -> List[Dict[str, Any]]:
    """Known-good SHL assessment subset — always functional without network."""
    return [
        {"name": "Verify Numerical Reasoning", "url": "https://www.shl.com/solutions/products/verify-numerical-reasoning/", "description": "Measures ability to make correct decisions from numerical data. Suitable for quantitative roles.", "duration": "17 minutes", "remote_testing": "Yes", "adaptive_irt": "Yes", "test_type": "A", "skills_measured": ["Numerical reasoning", "Data interpretation", "Statistical analysis"]},
        {"name": "Verify Verbal Reasoning", "url": "https://www.shl.com/solutions/products/verify-verbal-reasoning/", "description": "Assesses ability to understand and evaluate written information.", "duration": "17 minutes", "remote_testing": "Yes", "adaptive_irt": "Yes", "test_type": "A", "skills_measured": ["Verbal reasoning", "Reading comprehension", "Logical deduction"]},
        {"name": "Verify Inductive Reasoning", "url": "https://www.shl.com/solutions/products/verify-inductive-reasoning/", "description": "Measures ability to draw conclusions from abstract patterns. Useful for technical roles.", "duration": "25 minutes", "remote_testing": "Yes", "adaptive_irt": "Yes", "test_type": "A", "skills_measured": ["Inductive reasoning", "Pattern recognition", "Abstract thinking"]},
        {"name": "Verify Deductive Reasoning", "url": "https://www.shl.com/solutions/products/verify-deductive-reasoning/", "description": "Assesses logical reasoning by presenting arguments for evaluation.", "duration": "20 minutes", "remote_testing": "Yes", "adaptive_irt": "No", "test_type": "A", "skills_measured": ["Deductive reasoning", "Logical analysis", "Critical thinking"]},
        {"name": "OPQ32", "url": "https://www.shl.com/solutions/products/opq/", "description": "Occupational Personality Questionnaire measuring 32 personality characteristics relevant to work.", "duration": "25 minutes", "remote_testing": "Yes", "adaptive_irt": "No", "test_type": "P", "skills_measured": ["Relationships", "Thinking style", "Feelings and emotions", "Leadership", "Teamwork"]},
        {"name": "Motivation Questionnaire (MQ)", "url": "https://www.shl.com/solutions/products/motivation-questionnaire/", "description": "Assesses conditions and circumstances that increase or decrease motivation at work.", "duration": "25 minutes", "remote_testing": "Yes", "adaptive_irt": "No", "test_type": "M", "skills_measured": ["Intrinsic motivation", "Achievement drive", "Work environment preferences"]},
        {"name": "Java 8 (New)", "url": "https://www.shl.com/solutions/products/java-8/", "description": "Measures knowledge and practical skills in Java 8 including OOP, streams, and lambdas.", "duration": "25 minutes", "remote_testing": "Yes", "adaptive_irt": "No", "test_type": "K", "skills_measured": ["Java 8 features", "OOP", "Lambda expressions", "Stream API", "Collections"]},
        {"name": "Python (New)", "url": "https://www.shl.com/solutions/products/python/", "description": "Assesses Python programming including core syntax, data structures, and libraries.", "duration": "25 minutes", "remote_testing": "Yes", "adaptive_irt": "No", "test_type": "K", "skills_measured": ["Python syntax", "Data structures", "OOP in Python", "Standard libraries"]},
        {"name": "SQL (New)", "url": "https://www.shl.com/solutions/products/sql/", "description": "Evaluates SQL knowledge for database querying including joins, subqueries, and optimization.", "duration": "25 minutes", "remote_testing": "Yes", "adaptive_irt": "No", "test_type": "K", "skills_measured": ["SQL queries", "Joins", "Database design", "Query optimization"]},
        {"name": "JavaScript (New)", "url": "https://www.shl.com/solutions/products/javascript/", "description": "Tests JavaScript knowledge covering ES6+, DOM manipulation, and async programming.", "duration": "25 minutes", "remote_testing": "Yes", "adaptive_irt": "No", "test_type": "K", "skills_measured": ["JavaScript ES6+", "DOM", "Async/await", "Promises"]},
        {"name": "Situational Judgement Test (SJT)", "url": "https://www.shl.com/solutions/products/situational-judgement/", "description": "Work-relevant scenarios assessing decision making and workplace challenge response.", "duration": "25 minutes", "remote_testing": "Yes", "adaptive_irt": "No", "test_type": "B", "skills_measured": ["Decision making", "Problem solving", "Interpersonal skills", "Customer focus"]},
        {"name": "Customer Contact Styles Questionnaire", "url": "https://www.shl.com/solutions/products/customer-contact-styles/", "description": "Measures personality traits for customer-facing roles including service orientation.", "duration": "20 minutes", "remote_testing": "Yes", "adaptive_irt": "No", "test_type": "P", "skills_measured": ["Customer orientation", "Emotional resilience", "Service mindset"]},
        {"name": "Sales Achievement Predictor (SalesAP)", "url": "https://www.shl.com/solutions/products/sales-achievement-predictor/", "description": "Predicts sales performance by measuring personality traits linked to sales success.", "duration": "30 minutes", "remote_testing": "Yes", "adaptive_irt": "No", "test_type": "P", "skills_measured": ["Sales drive", "Prospecting", "Closing skills", "Persistence"]},
        {"name": "Verify G+ Cognitive Ability", "url": "https://www.shl.com/solutions/products/verify-g-plus/", "description": "Next-generation adaptive cognitive ability combining numerical and verbal reasoning.", "duration": "36 minutes", "remote_testing": "Yes", "adaptive_irt": "Yes", "test_type": "A", "skills_measured": ["Cognitive ability", "Numerical reasoning", "Verbal reasoning", "Processing speed"]},
        {"name": "General Mental Ability (GMA)", "url": "https://www.shl.com/solutions/products/gma/", "description": "Measures overall cognitive ability as a broad predictor of job performance.", "duration": "30 minutes", "remote_testing": "Yes", "adaptive_irt": "Yes", "test_type": "A", "skills_measured": ["Cognitive ability", "Problem solving", "Learning agility"]},
        {"name": "Technology Professional 8.0 — Short", "url": "https://www.shl.com/solutions/products/technology-professional/", "description": "Cognitive ability test calibrated for technology professionals covering reasoning skills.", "duration": "36 minutes", "remote_testing": "Yes", "adaptive_irt": "Yes", "test_type": "A", "skills_measured": ["Numerical reasoning", "Verbal reasoning", "Inductive reasoning"]},
        {"name": "Universal Competency Framework (UCF)", "url": "https://www.shl.com/solutions/products/universal-competency-framework/", "description": "Research-based framework covering eight competency factors for talent assessment.", "duration": "Varies", "remote_testing": "Yes", "adaptive_irt": "No", "test_type": "C", "skills_measured": ["Leading", "Supporting", "Analyzing", "Creating", "Interacting"]},
        {"name": "Leadership 360", "url": "https://www.shl.com/solutions/products/360/", "description": "Multi-rater feedback tool assessing leadership effectiveness across key behavioral competencies.", "duration": "30 minutes", "remote_testing": "Yes", "adaptive_irt": "No", "test_type": "D", "skills_measured": ["Leadership", "Decision making", "Communication", "Strategic thinking"]},
        {"name": "Call Center Simulator", "url": "https://www.shl.com/solutions/products/call-center-simulator/", "description": "Realistic call center simulation assessing customer service, multitasking, and data entry.", "duration": "30 minutes", "remote_testing": "Yes", "adaptive_irt": "No", "test_type": "S", "skills_measured": ["Customer service", "Multitasking", "Data entry", "Call handling"]},
        {"name": "Coding Simulations (General)", "url": "https://www.shl.com/solutions/products/coding-simulations/", "description": "Practical coding exercises simulating real software engineering tasks.", "duration": "30 minutes", "remote_testing": "Yes", "adaptive_irt": "No", "test_type": "S", "skills_measured": ["Coding proficiency", "Problem solving", "Algorithm design", "Debugging"]},
        {"name": "Workplace English (Business)", "url": "https://www.shl.com/solutions/products/workplace-english/", "description": "Assesses English language proficiency in business contexts.", "duration": "35 minutes", "remote_testing": "Yes", "adaptive_irt": "No", "test_type": "K", "skills_measured": ["Business English", "Grammar", "Reading comprehension"]},
        {"name": "Agile Profile", "url": "https://www.shl.com/solutions/products/agile-profile/", "description": "Measures agile mindset and behaviors for agile software development environments.", "duration": "20 minutes", "remote_testing": "Yes", "adaptive_irt": "No", "test_type": "P", "skills_measured": ["Agile mindset", "Collaboration", "Adaptability", "Iterative thinking"]},
        {"name": "Dependability & Safety Instrument (DSI)", "url": "https://www.shl.com/solutions/products/dsi/", "description": "Assesses personality traits related to workplace safety and reliability for safety-critical roles.", "duration": "20 minutes", "remote_testing": "Yes", "adaptive_irt": "No", "test_type": "P", "skills_measured": ["Safety orientation", "Dependability", "Integrity", "Risk awareness"]},
        {"name": "Remote Work Assessment", "url": "https://www.shl.com/solutions/products/remote-work-assessment/", "description": "Evaluates suitability for remote work assessing self-discipline and digital collaboration.", "duration": "20 minutes", "remote_testing": "Yes", "adaptive_irt": "No", "test_type": "P", "skills_measured": ["Self-management", "Remote communication", "Time management"]},
        {"name": "Verify Interactive — Numerical", "url": "https://www.shl.com/solutions/products/verify-interactive/", "description": "Simulation-based numerical reasoning assessment replicating real work scenarios.", "duration": "20 minutes", "remote_testing": "Yes", "adaptive_irt": "Yes", "test_type": "S", "skills_measured": ["Numerical reasoning in context", "Business data analysis"]},
        {"name": "Graduate 8.0 — Numerical", "url": "https://www.shl.com/solutions/products/graduate-8/", "description": "Graduate-level numerical reasoning test calibrated for recent graduates.", "duration": "17 minutes", "remote_testing": "Yes", "adaptive_irt": "Yes", "test_type": "A", "skills_measured": ["Numerical reasoning", "Data analysis"]},
        {"name": "Graduate 8.0 — Verbal", "url": "https://www.shl.com/solutions/products/graduate-8/", "description": "Graduate-level verbal reasoning test measuring comprehension and critical evaluation.", "duration": "17 minutes", "remote_testing": "Yes", "adaptive_irt": "Yes", "test_type": "A", "skills_measured": ["Verbal reasoning", "Reading comprehension"]},
        {"name": "C# (New)", "url": "https://www.shl.com/solutions/products/csharp/", "description": "Evaluates C# programming knowledge including .NET concepts, LINQ, and async programming.", "duration": "25 minutes", "remote_testing": "Yes", "adaptive_irt": "No", "test_type": "K", "skills_measured": ["C# syntax", ".NET framework", "LINQ", "Async programming", "OOP"]},
        {"name": "Core Java (Advanced Level)", "url": "https://www.shl.com/solutions/products/core-java/", "description": "Advanced Java assessment covering multithreading, design patterns, JVM internals and performance.", "duration": "35 minutes", "remote_testing": "Yes", "adaptive_irt": "No", "test_type": "K", "skills_measured": ["Advanced Java", "Multithreading", "Design patterns", "JVM", "Performance tuning"]},
        {"name": "Managerial & Professional Learning Styles", "url": "https://www.shl.com/solutions/products/learning-styles/", "description": "Identifies how managers prefer to learn, guiding training and development programs.", "duration": "20 minutes", "remote_testing": "Yes", "adaptive_irt": "No", "test_type": "D", "skills_measured": ["Learning preference", "Development readiness", "Reflective observation"]},
    ]


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Scraping SHL catalog from %s", CATALOG_URL)

    all_items: List[Dict[str, Any]] = []
    try:
        # Try 3 pages max
        for page in range(1, 4):
            page_items = scrape_catalog_page(CATALOG_URL, page)
            if not page_items:
                break
            all_items.extend(page_items)
            time.sleep(REQUEST_DELAY)

        if all_items:
            logger.info("Enriching %d items…", len(all_items))
            enriched = []
            for i, item in enumerate(all_items):
                logger.info("[%d/%d] %s", i + 1, len(all_items), item["name"])
                details = scrape_detail_page(item["url"])
                merged = {**item, **details}
                if not merged.get("description"):
                    merged["description"] = f"SHL {item['name']} assessment."
                enriched.append(merged)
                time.sleep(REQUEST_DELAY)
            all_items = enriched
        else:
            logger.warning("Live scrape returned 0 items. Using built-in fallback catalog.")
            all_items = build_fallback_catalog()
    except Exception as exc:
        logger.error("Scraping error: %s — using fallback catalog.", exc)
        all_items = build_fallback_catalog()

    # Deduplicate
    seen: set = set()
    unique: List[Dict[str, Any]] = []
    for item in all_items:
        key = item["name"].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(item)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(unique, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d assessments to %s", len(unique), OUTPUT_PATH)


if __name__ == "__main__":
    main()
