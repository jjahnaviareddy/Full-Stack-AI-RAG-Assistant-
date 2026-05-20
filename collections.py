"""
Collections API — manage knowledge base collections.
"""
from datetime import datetime

from fastapi import APIRouter, HTTPException

from app.models.schemas import CollectionCreate, CollectionListResponse, CollectionResponse
from app.services.rag_service import get_rag_service

router = APIRouter(prefix="/collections", tags=["Collections"])


@router.get("/", response_model=CollectionListResponse)
async def list_collections():
    """List all knowledge base collections."""
    rag = get_rag_service()
    cols = rag.list_collections()
    responses = [
        CollectionResponse(
            id=c["id"],
            name=c["name"],
            description=c.get("description", ""),
            document_count=c["document_count"],
            created_at=datetime.utcnow(),
        )
        for c in cols
    ]
    return CollectionListResponse(collections=responses, total=len(responses))


@router.post("/", response_model=CollectionResponse, status_code=201)
async def create_collection(body: CollectionCreate):
    """Create a new collection (knowledge base)."""
    rag = get_rag_service()
    # Just register the name in the registry
    col_id = body.name.lower().replace(" ", "-")
    if col_id not in rag._document_registry:
        rag._document_registry[col_id] = []
        rag._save_registry()
    return CollectionResponse(
        id=col_id,
        name=body.name,
        description=body.description,
        document_count=0,
        created_at=datetime.utcnow(),
    )
