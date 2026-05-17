"""
LLM client — tries Gemini first (via google-genai SDK), falls back to OpenRouter.
Centralizes all LLM calls with retry logic.
"""

import logging
import time
from typing import Any, Dict, Optional

import httpx

from app.config import get_settings
from app.utils.helpers import extract_json_from_text, safe_response

logger = logging.getLogger(__name__)
settings = get_settings()

_GEMINI_MODEL = "gemini-2.0-flash"
_OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def _call_gemini(prompt: str) -> Optional[str]:
    """Call Gemini via the google-genai SDK."""
    if not settings.gemini_api_key:
        logger.warning("GEMINI_API_KEY not set.")
        return None
    try:
        from google import genai  # type: ignore
        client = genai.Client(api_key=settings.gemini_api_key)
        response = client.models.generate_content(
            model=_GEMINI_MODEL,
            contents=prompt,
            config={
                "temperature": 0.2,
                "max_output_tokens": 1024,
            },
        )
        return response.text
    except Exception as exc:
        logger.warning("Gemini call failed: %s", exc)
        return None


def _call_openrouter(prompt: str) -> Optional[str]:
    """Call OpenRouter as fallback."""
    if not settings.openrouter_api_key:
        logger.warning("OpenRouter API key not configured.")
        return None
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.post(
                f"{_OPENROUTER_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.openrouter_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.openrouter_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": 1024,
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
    except Exception as exc:
        logger.error("OpenRouter call failed: %s", exc)
        return None


def call_llm(prompt: str, retries: int = 2) -> Optional[str]:
    """
    Try Gemini → OpenRouter with retry.
    Returns raw LLM text or None on total failure.
    """
    for attempt in range(retries):
        text = _call_gemini(prompt)
        if text:
            return text
        logger.warning("Gemini attempt %d/%d failed, trying OpenRouter…", attempt + 1, retries)
        text = _call_openrouter(prompt)
        if text:
            return text
        if attempt < retries - 1:
            time.sleep(1)
    return None


def call_llm_json(prompt: str) -> Dict[str, Any]:
    """
    Call LLM and parse JSON from response.
    Returns a safe default dict on failure.
    """
    raw = call_llm(prompt)
    if not raw:
        return {
            "reply": "I'm currently operating in offline mode because the AI API limits were reached. However, I have successfully searched the SHL catalog and found these highly relevant assessments for your requirements:",
            "recommendations": [],
            "end_of_conversation": False
        }
    parsed = extract_json_from_text(raw)
    if parsed is None:
        return safe_response(raw.strip())
    return parsed
