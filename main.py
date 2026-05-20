"""
RAG Assistant — FastAPI Application Entry Point
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from app.api import chat, collections, documents
from app.core.config import get_settings
from app.models.schemas import HealthResponse
from app.services.rag_service import get_rag_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup / shutdown."""
    settings = get_settings()
    logger.info(f"🚀 Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"   OpenAI model  : {settings.openai_model}")
    logger.info(f"   Embedding model: {settings.openai_embedding_model}")
    logger.info(f"   FAISS index    : {settings.faiss_index_path}")
    logger.info(f"   API key set    : {'✅' if settings.openai_api_key else '❌'}")

    # Pre-warm RAG service (loads existing indexes)
    get_rag_service()

    yield

    logger.info("👋 Shutting down RAG Assistant")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Full-Stack AI RAG Assistant — LangChain + FAISS + FastAPI",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────────────
    prefix = settings.api_prefix
    app.include_router(documents.router, prefix=prefix)
    app.include_router(chat.router, prefix=prefix)
    app.include_router(collections.router, prefix=prefix)

    # ── Health Check ──────────────────────────────────────────────────────────
    @app.get(f"{prefix}/health", response_model=HealthResponse, tags=["Health"])
    async def health():
        rag = get_rag_service()
        stats = rag.get_stats()
        return HealthResponse(
            status="ok",
            version=settings.app_version,
            openai_configured=bool(settings.openai_api_key),
            **stats,
        )

    @app.get("/", include_in_schema=False)
    async def root():
        return JSONResponse({"message": "RAG Assistant API", "docs": "/docs"})

    return app


app = create_app()
