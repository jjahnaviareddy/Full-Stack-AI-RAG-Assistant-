"""
Pydantic models for request/response validation.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ─── Document Models ──────────────────────────────────────────────────────────

class DocumentMetadata(BaseModel):
    filename: str
    file_type: str
    file_size: int
    collection_id: str = "default"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    chunk_count: int = 0
    extra: Dict[str, Any] = Field(default_factory=dict)


class DocumentResponse(BaseModel):
    id: str
    filename: str
    file_type: str
    file_size: int
    collection_id: str
    chunk_count: int
    created_at: datetime
    status: str = "ready"


class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    total: int


# ─── Collection Models ────────────────────────────────────────────────────────

class CollectionCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(default="", max_length=500)


class CollectionResponse(BaseModel):
    id: str
    name: str
    description: str
    document_count: int
    created_at: datetime


class CollectionListResponse(BaseModel):
    collections: List[CollectionResponse]
    total: int


# ─── Chat / Query Models ──────────────────────────────────────────────────────

class SourceDocument(BaseModel):
    content: str
    metadata: Dict[str, Any]
    score: float


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    collection_id: str = Field(default="default")
    top_k: int = Field(default=4, ge=1, le=10)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    include_sources: bool = True


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument] = Field(default_factory=list)
    collection_id: str
    question: str
    model: str
    tokens_used: Optional[int] = None
    latency_ms: Optional[float] = None


class StreamQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    collection_id: str = Field(default="default")
    top_k: int = Field(default=4, ge=1, le=10)


# ─── Health Models ────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    openai_configured: bool
    collections_count: int
    documents_count: int


# ─── Error Models ─────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None
