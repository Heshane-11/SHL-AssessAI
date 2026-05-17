"""
FastAPI application entry point.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routes.chat import router
from app.services.retriever import retriever
from app.utils.helpers import setup_logging

settings = get_settings()
setup_logging("INFO")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up the FAISS index and embedding model on startup."""
    logger.info("Starting SHL Assessment Recommender Agent…")
    if retriever.is_ready:
        logger.info("FAISS index loaded and ready.")
    else:
        logger.warning(
            "FAISS index not available. Run scripts/build_index.py to enable semantic search."
        )
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="SHL Assessment Recommender API",
    description=(
        "Conversational agent that recommends SHL assessments based on hiring requirements. "
        "Powered by Gemini + FAISS semantic search."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for development; restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
