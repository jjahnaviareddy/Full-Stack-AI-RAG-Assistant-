"""
Documents API — upload, list, delete documents.
"""
import uuid
from pathlib import Path

import aiofiles
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from loguru import logger

from app.core.config import get_settings
from app.models.schemas import DocumentListResponse, DocumentResponse
from app.services.rag_service import get_rag_service
from app.utils.document_loader import is_supported

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/upload", response_model=DocumentResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    collection_id: str = Form(default="default"),
):
    """
    Upload a document (PDF, TXT, DOCX, CSV) and index it into a collection.
    """
    settings = get_settings()

    # Validate file type
    if not is_supported(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported: PDF, TXT, DOCX, CSV, MD",
        )

    # Validate file size
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.max_upload_size_mb:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Max: {settings.max_upload_size_mb} MB",
        )

    # Save to disk
    document_id = str(uuid.uuid4())
    safe_filename = f"{document_id}_{file.filename}"
    file_path = Path(settings.upload_dir) / safe_filename

    async with aiofiles.open(file_path, "wb") as f:
        await f.write(content)

    # Ingest into vector store
    try:
        rag = get_rag_service()
        chunk_count = await rag.ingest_document(
            file_path=str(file_path),
            filename=file.filename,
            document_id=document_id,
            collection_id=collection_id,
        )
    except Exception as e:
        # Clean up on failure
        file_path.unlink(missing_ok=True)
        logger.error(f"Ingestion failed for '{file.filename}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

    from datetime import datetime
    return DocumentResponse(
        id=document_id,
        filename=file.filename,
        file_type=Path(file.filename).suffix.lower(),
        file_size=len(content),
        collection_id=collection_id,
        chunk_count=chunk_count,
        created_at=datetime.utcnow(),
        status="ready",
    )


@router.get("/", response_model=DocumentListResponse)
async def list_documents(collection_id: str = None):
    """List all uploaded documents, optionally filtered by collection."""
    rag = get_rag_service()
    docs = rag.list_documents(collection_id)

    from datetime import datetime
    doc_responses = [
        DocumentResponse(
            id=d["document_id"],
            filename=d["filename"],
            file_type=Path(d["filename"]).suffix.lower(),
            file_size=0,
            collection_id=collection_id or "default",
            chunk_count=d.get("chunks", 0),
            created_at=datetime.utcnow(),
            status="ready",
        )
        for d in docs
    ]
    return DocumentListResponse(documents=doc_responses, total=len(doc_responses))


@router.delete("/{document_id}", status_code=204)
async def delete_document(document_id: str, collection_id: str = "default"):
    """Delete a document from a collection."""
    rag = get_rag_service()
    deleted = rag.delete_document(document_id, collection_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
