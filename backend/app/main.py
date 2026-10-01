"""
Application entrypoint.

Original Phase 1 wiring is unchanged (the /health route and create_app()
structure). The only additions are:
  1. CORS middleware, so the static frontend can call the API.
  2. Mounting the `analyze` router (backend/app/routes/analyze.py), which is
     purely additive and does not alter any extraction, RAG, RASE, or
     RuleEngine logic.
  3. Serving the frontend/ static site at "/", so the whole app runs from a
     single `uvicorn backend.app.main:app` process.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import get_settings
from backend.tools.logging_config import get_logger

logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = REPO_ROOT / "frontend"


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        description="Pre-submission advisory system for Indian municipal building regulations.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "environment": settings.environment}

    from backend.app.routes.analyze import router as analyze_router
    from backend.app.routes.regulations import router as regulations_router

    app.include_router(analyze_router)
    app.include_router(regulations_router)

    if FRONTEND_DIR.exists():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

    logger.info("app initialized", extra={"environment": settings.environment})
    return app


app = create_app()
