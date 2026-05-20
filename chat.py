"""
Chat API — query the RAG pipeline (regular + streaming).
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.core.config import get_settings
from app.models.schemas import QueryRequest, QueryResponse, StreamQueryRequest
from app.services.rag_service import get_rag_service

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Ask a question over the uploaded documents.
    Returns a complete answer with source citations.
    """
    settings = get_settings()

    if not settings.openai_api_key:
        raise HTTPException(
            status_code=503,
            detail="OpenAI API key not configured. Set OPENAI_API_KEY in .env",
        )

    try:
        rag = get_rag_service()
        result = await rag.query(
            question=request.question,
            collection_id=request.collection_id,
            top_k=min(request.top_k, settings.max_top_k),
            temperature=request.temperature,
            include_sources=request.include_sources,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.post("/stream")
async def stream_query(request: StreamQueryRequest):
    """
    Streaming version of /query. Returns server-sent events (SSE).
    Each event is a JSON object with 'type' field:
      - {"type": "sources", "sources": [...]}
      - {"type": "token", "content": "..."}
      - {"type": "done"}
    """
    settings = get_settings()

    if not settings.openai_api_key:
        raise HTTPException(
            status_code=503,
            detail="OpenAI API key not configured.",
        )

    rag = get_rag_service()

    return StreamingResponse(
        rag.stream_query(
            question=request.question,
            collection_id=request.collection_id,
            top_k=min(request.top_k, settings.max_top_k),
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
