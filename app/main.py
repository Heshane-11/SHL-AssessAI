"""
FastAPI application entry point.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.config import get_settings
from app.routes.chat import router
from app.services.retriever import retriever
from app.utils.helpers import setup_logging

settings = get_settings()
setup_logging("INFO")
logger = logging.getLogger(__name__)

LANDING_PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>SHL AssessAI — Assessment Recommender</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap" rel="stylesheet"/>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', sans-serif;
      background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #fff;
      overflow: hidden;
    }
    .bg-orbs {
      position: fixed; inset: 0; z-index: 0; pointer-events: none;
    }
    .orb {
      position: absolute; border-radius: 50%; filter: blur(80px); opacity: 0.35;
      animation: drift 10s ease-in-out infinite alternate;
    }
    .orb1 { width: 500px; height: 500px; background: #7c3aed; top: -150px; left: -100px; animation-delay: 0s; }
    .orb2 { width: 400px; height: 400px; background: #2563eb; bottom: -100px; right: -80px; animation-delay: 3s; }
    .orb3 { width: 300px; height: 300px; background: #06b6d4; top: 40%; left: 60%; animation-delay: 6s; }
    @keyframes drift {
      from { transform: translate(0, 0) scale(1); }
      to   { transform: translate(30px, 20px) scale(1.05); }
    }
    .card {
      position: relative; z-index: 1;
      background: rgba(255,255,255,0.06);
      backdrop-filter: blur(24px);
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: 28px;
      padding: 56px 48px;
      max-width: 680px; width: 90%;
      text-align: center;
      box-shadow: 0 32px 80px rgba(0,0,0,0.4);
      animation: fadeUp 0.8s ease both;
    }
    @keyframes fadeUp {
      from { opacity: 0; transform: translateY(30px); }
      to   { opacity: 1; transform: translateY(0); }
    }
    .badge {
      display: inline-block;
      background: linear-gradient(90deg, #7c3aed, #2563eb);
      color: #fff; font-size: 11px; font-weight: 700;
      letter-spacing: 2px; text-transform: uppercase;
      padding: 6px 18px; border-radius: 999px;
      margin-bottom: 24px;
    }
    h1 {
      font-size: clamp(2rem, 5vw, 3rem);
      font-weight: 800;
      line-height: 1.1;
      margin-bottom: 16px;
      background: linear-gradient(90deg, #c4b5fd, #93c5fd, #67e8f9);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    p.subtitle {
      color: rgba(255,255,255,0.65);
      font-size: 1rem;
      line-height: 1.7;
      margin-bottom: 40px;
    }
    .chips {
      display: flex; flex-wrap: wrap; justify-content: center; gap: 10px;
      margin-bottom: 40px;
    }
    .chip {
      background: rgba(255,255,255,0.08);
      border: 1px solid rgba(255,255,255,0.15);
      border-radius: 999px;
      padding: 6px 16px;
      font-size: 12px; color: rgba(255,255,255,0.75);
      font-weight: 500;
    }
    .actions { display: flex; flex-direction: column; gap: 14px; }
    .btn {
      display: flex; align-items: center; justify-content: center; gap: 10px;
      padding: 16px 28px; border-radius: 14px;
      font-size: 15px; font-weight: 600;
      text-decoration: none; cursor: pointer;
      transition: transform 0.2s, box-shadow 0.2s;
      border: none;
    }
    .btn:hover { transform: translateY(-3px); box-shadow: 0 12px 32px rgba(0,0,0,0.3); }
    .btn-primary {
      background: linear-gradient(135deg, #7c3aed, #2563eb);
      color: #fff;
      box-shadow: 0 4px 20px rgba(124,58,237,0.4);
    }
    .btn-secondary {
      background: rgba(255,255,255,0.08);
      border: 1px solid rgba(255,255,255,0.18);
      color: #fff;
    }
    .btn-success {
      background: rgba(16,185,129,0.15);
      border: 1px solid rgba(16,185,129,0.4);
      color: #6ee7b7;
    }
    .status-bar {
      margin-top: 36px;
      display: flex; align-items: center; justify-content: center; gap: 8px;
      font-size: 13px; color: rgba(255,255,255,0.45);
    }
    .status-dot {
      width: 8px; height: 8px; border-radius: 50%;
      background: #10b981;
      box-shadow: 0 0 8px #10b981;
      animation: pulse 1.5s ease-in-out infinite;
    }
    @keyframes pulse {
      0%, 100% { opacity: 1; } 50% { opacity: 0.3; }
    }
  </style>
</head>
<body>
  <div class="bg-orbs">
    <div class="orb orb1"></div>
    <div class="orb orb2"></div>
    <div class="orb orb3"></div>
  </div>

  <div class="card">
    <div class="badge">⚡ Powered by FAISS + Gemini</div>
    <h1>SHL AssessAI</h1>
    <p class="subtitle">
      An enterprise-grade conversational AI agent that recommends, compares,
      and refines SHL assessments based on your hiring requirements.
    </p>

    <div class="chips">
      <span class="chip">🛡️ Prompt Injection Guard</span>
      <span class="chip">🧠 Hallucination Prevention</span>
      <span class="chip">🔋 Offline Fallback Mode</span>
      <span class="chip">📡 Stateless API</span>
    </div>

    <div class="actions">
      <a href="/docs" class="btn btn-primary" id="swagger-btn">
        📖 &nbsp; Open Interactive API Docs (Swagger)
      </a>
      <a href="/redoc" class="btn btn-secondary" id="redoc-btn">
        📄 &nbsp; View ReDoc Documentation
      </a>
      <a href="/health" class="btn btn-success" id="health-btn">
        💚 &nbsp; Check API Health Status
      </a>
    </div>

    <div class="status-bar">
      <div class="status-dot"></div>
      API is live and running &nbsp;·&nbsp; v1.0.0
    </div>
  </div>
</body>
</html>
"""


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


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def landing_page():
    """Beautiful landing page with links to docs, redoc, and health."""
    return HTMLResponse(content=LANDING_PAGE_HTML)


app.include_router(router)
