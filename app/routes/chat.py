"""
API routes — /health and /chat endpoints.
"""

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from app.models.schemas import ChatRequest, ChatResponse, HealthResponse
from app.services.agent import handle_chat

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["Health"],
)
async def health_check() -> HealthResponse:
    """Returns a simple OK status to confirm the service is running."""
    return HealthResponse(status="ok")


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Conversational SHL assessment recommender",
    tags=["Chat"],
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Accepts conversation history and returns:
    - A conversational reply
    - 0–10 SHL assessment recommendations
    - end_of_conversation flag
    """
    try:
        response = handle_chat(request)
        return response
    except Exception as exc:
        logger.exception("Unhandled error in /chat: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred. Please try again.",
        )
