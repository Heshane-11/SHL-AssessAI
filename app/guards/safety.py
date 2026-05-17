"""
Guardrails: detect prompt injection, out-of-scope requests,
and hallucinated recommendations.
"""

import re
from typing import Tuple

# ── Injection patterns ──────────────────────────────────────────────────────
_INJECTION_PATTERNS = [
    r"ignore (previous|all|your) (instructions?|prompts?|rules?|constraints?)",
    r"you are now",
    r"forget (everything|all|your instructions?)",
    r"act as (a|an|the)?\s+(?!shl)",  # "act as X" where X isn't SHL
    r"pretend (you are|to be)",
    r"disregard (your|all|the) (system|previous|instructions?)",
    r"reveal (your|the) (system|hidden|secret) prompt",
    r"override (your|the) (instructions?|rules?|constraints?)",
    r"do anything now",
    r"jailbreak",
    r"developer mode",
]

# ── Out-of-scope topics ──────────────────────────────────────────────────────
_OUT_OF_SCOPE_PATTERNS = [
    r"\b(legal advice|employment law|discrimination|lawsuit|sue)\b",
    r"\b(salary|pay scale|compensation|offer letter|how much (to pay|should i pay|do i pay))\b",
    r"\b(interview tips|resume review|cv review|general hiring advice)\b",
    r"\b(write (me|my|a) (code|script|essay|email))\b",
    r"\b(stock (price|market)|cryptocurrency|bitcoin|invest)\b",
    r"\b(weather|news|sports|entertainment)\b",
    r"\b(recipe|cook|food|restaurant)\b",
]

_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]
_SCOPE_RE = [re.compile(p, re.IGNORECASE) for p in _OUT_OF_SCOPE_PATTERNS]


def check_injection(text: str) -> Tuple[bool, str]:
    """
    Returns (is_injection, reason).
    """
    for pattern in _INJECTION_RE:
        if pattern.search(text):
            return True, "prompt_injection"
    return False, ""


def check_out_of_scope(text: str) -> Tuple[bool, str]:
    """
    Returns (is_out_of_scope, reason).
    """
    for pattern in _SCOPE_RE:
        if pattern.search(text):
            return True, "out_of_scope"
    return False, ""


def validate_recommendations(
    recommendations: list, catalog_names: set
) -> list:
    """
    Filter out any recommendations that are NOT in the catalog.
    Prevents hallucinated assessments from leaking into the response.
    """
    validated = []
    for rec in recommendations:
        name = rec.get("name", "")
        # Check exact match or close match (strip whitespace/case)
        if any(name.lower().strip() == cn.lower().strip() for cn in catalog_names):
            validated.append(rec)
    return validated


def run_all_guards(text: str) -> Tuple[bool, str]:
    """
    Run all guards on user input.
    Returns (blocked, reason) — reason is '' if not blocked.
    """
    is_inject, reason = check_injection(text)
    if is_inject:
        return True, reason

    is_oos, reason = check_out_of_scope(text)
    if is_oos:
        return True, reason

    return False, ""
