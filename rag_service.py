"""
RAG (Retrieval-Augmented Generation) Service.

Handles:
- Document ingestion + chunking
- FAISS vector index management
- Semantic similarity search
- LLM-powered answer generation with source citations
"""
import json
import time
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Tuple

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from loguru import logger

from app.core.config import get_settings
from app.models.schemas import QueryResponse, SourceDocument
from app.utils.document_loader import load_document


# ─── System Prompt ────────────────────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on the provided context documents.

Instructions:
- Answer ONLY based on the provided context. Do not use outside knowledge.
- If the context doesn't contain enough information, say "I don't have enough information in the provided documents to answer this question."
- Be concise, accurate, and helpful.
- When referencing information, mention the source document when possible.
- Format your answer clearly. Use bullet points or numbered lists when appropriate.

Context:
{context}
"""

RAG_HUMAN_PROMPT = "Question: {question}"


class RAGService:
    """
    Main RAG service that manages vector stores and LLM interactions.
    Each 'collection' is a separate FAISS index stored on disk.
    """

    def __init__(self):
        self.settings = get_settings()
        self._embeddings: Optional[OpenAIEmbeddings] = None
        self._llm: Optional[ChatOpenAI] = None
        self._vector_stores: Dict[str, FAISS] = {}
        self._document_registry: Dict[str, List[dict]] = {}  # collection_id -> [doc_metadata]
        self._load_registry()

    # ─── Private: Lazy Init ───────────────────────────────────────────────────

    @property
    def embeddings(self) -> OpenAIEmbeddings:
        if self._embeddings is None:
            self._embeddings = OpenAIEmbeddings(
                model=self.settings.openai_embedding_model,
                openai_api_key=self.settings.openai_api_key,
            )
        return self._embeddings

    @property
    def llm(self) -> ChatOpenAI:
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=self.settings.openai_model,
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens,
                openai_api_key=self.settings.openai_api_key,
                streaming=True,
            )
        return self._llm

    # ─── Registry (persist doc metadata to disk) ──────────────────────────────

    def _registry_path(self) -> Path:
        return Path(self.settings.faiss_index_path) / "registry.json"

    def _load_registry(self) -> None:
        path = self._registry_path()
        if path.exists():
            with open(path, "r") as f:
                self._document_registry = json.load(f)
        else:
            self._document_registry = {}

    def _save_registry(self) -> None:
        with open(self._registry_path(), "w") as f:
            json.dump(self._document_registry, f, indent=2, default=str)

    # ─── Vector Store Management ──────────────────────────────────────────────

    def _index_path(self, collection_id: str) -> str:
        return str(Path(self.settings.faiss_index_path) / collection_id)

    def _load_vector_store(self, collection_id: str) -> Optional[FAISS]:
        """Load FAISS index from disk if it exists."""
        if collection_id in self._vector_stores:
            return self._vector_stores[collection_id]

        index_path = self._index_path(collection_id)
        if Path(index_path).exists():
            try:
                vs = FAISS.load_local(
                    index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                self._vector_stores[collection_id] = vs
                logger.info(f"Loaded FAISS index for collection '{collection_id}'")
                return vs
            except Exception as e:
                logger.warning(f"Failed to load index for '{collection_id}': {e}")
        return None

    def _save_vector_store(self, collection_id: str, vs: FAISS) -> None:
        """Persist FAISS index to disk."""
        index_path = self._index_path(collection_id)
        Path(index_path).mkdir(parents=True, exist_ok=True)
        vs.save_local(index_path)
        self._vector_stores[collection_id] = vs
        logger.info(f"Saved FAISS index for collection '{collection_id}'")

    # ─── Document Ingestion ───────────────────────────────────────────────────

    async def ingest_document(
        self,
        file_path: str,
        filename: str,
        document_id: str,
        collection_id: str = "default",
    ) -> int:
        """
        Load, chunk, embed, and index a document.
        Returns the number of chunks created.
        """
        logger.info(f"Ingesting '{filename}' into collection '{collection_id}'")

        # 1. Load document
        raw_docs = load_document(file_path, filename)

        # 2. Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(raw_docs)

        # 3. Add document_id to chunk metadata
        for chunk in chunks:
            chunk.metadata["document_id"] = document_id
            chunk.metadata["collection_id"] = collection_id

        if not chunks:
            raise ValueError(f"No text content extracted from '{filename}'")

        # 4. Embed and index
        existing_vs = self._load_vector_store(collection_id)
        if existing_vs is None:
            vs = FAISS.from_documents(chunks, self.embeddings)
        else:
            vs = existing_vs
            vs.add_documents(chunks)

        self._save_vector_store(collection_id, vs)

        # 5. Register document
        if collection_id not in self._document_registry:
            self._document_registry[collection_id] = []
        self._document_registry[collection_id].append(
            {"document_id": document_id, "filename": filename, "chunks": len(chunks)}
        )
        self._save_registry()

        logger.info(f"Indexed {len(chunks)} chunks from '{filename}'")
        return len(chunks)

    # ─── Retrieval ────────────────────────────────────────────────────────────

    def retrieve(
        self, question: str, collection_id: str = "default", top_k: int = 4
    ) -> List[Tuple[Document, float]]:
        """Retrieve the most relevant document chunks for a question."""
        vs = self._load_vector_store(collection_id)
        if vs is None:
            return []

        results = vs.similarity_search_with_score(question, k=top_k)
        return results

    # ─── RAG Query (non-streaming) ────────────────────────────────────────────

    async def query(
        self,
        question: str,
        collection_id: str = "default",
        top_k: int = 4,
        temperature: Optional[float] = None,
        include_sources: bool = True,
    ) -> QueryResponse:
        """Full RAG pipeline: retrieve → augment → generate."""
        start = time.time()

        # 1. Retrieve
        retrieved = self.retrieve(question, collection_id, top_k)

        if not retrieved:
            return QueryResponse(
                answer="No documents found in this collection. Please upload documents first.",
                sources=[],
                collection_id=collection_id,
                question=question,
                model=self.settings.openai_model,
            )

        # 2. Build context
        context_parts = []
        sources = []
        for doc, score in retrieved:
            context_parts.append(
                f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}"
            )
            sources.append(
                SourceDocument(
                    content=doc.page_content,
                    metadata=doc.metadata,
                    score=float(score),
                )
            )
        context = "\n\n---\n\n".join(context_parts)

        # 3. Build prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            ("human", RAG_HUMAN_PROMPT),
        ])

        # 4. Call LLM
        llm = ChatOpenAI(
            model=self.settings.openai_model,
            temperature=temperature or self.settings.temperature,
            max_tokens=self.settings.max_tokens,
            openai_api_key=self.settings.openai_api_key,
        )

        chain = prompt | llm
        response = await chain.ainvoke({"context": context, "question": question})
        answer = response.content

        latency = (time.time() - start) * 1000

        return QueryResponse(
            answer=answer,
            sources=sources if include_sources else [],
            collection_id=collection_id,
            question=question,
            model=self.settings.openai_model,
            latency_ms=round(latency, 2),
        )

    # ─── RAG Query (streaming) ────────────────────────────────────────────────

    async def stream_query(
        self,
        question: str,
        collection_id: str = "default",
        top_k: int = 4,
    ) -> AsyncGenerator[str, None]:
        """Streaming RAG: yields text tokens as server-sent events."""
        retrieved = self.retrieve(question, collection_id, top_k)

        if not retrieved:
            yield "data: No documents found in this collection. Please upload documents first.\n\n"
            yield "data: [DONE]\n\n"
            return

        context = "\n\n---\n\n".join(
            f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}"
            for doc, _ in retrieved
        )

        # Yield source metadata first
        sources_data = [
            {
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page"),
                "score": round(float(score), 4),
            }
            for doc, score in retrieved
        ]
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources_data})}\n\n"

        prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            ("human", RAG_HUMAN_PROMPT),
        ])

        chain = prompt | self.llm

        async for chunk in chain.astream({"context": context, "question": question}):
            if chunk.content:
                yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    # ─── Document Management ──────────────────────────────────────────────────

    def list_documents(self, collection_id: Optional[str] = None) -> List[dict]:
        if collection_id:
            return self._document_registry.get(collection_id, [])
        all_docs = []
        for col_docs in self._document_registry.values():
            all_docs.extend(col_docs)
        return all_docs

    def delete_document(self, document_id: str, collection_id: str) -> bool:
        """
        Remove a document from the registry.
        Note: FAISS doesn't support single-doc deletion without full rebuild.
        For production, use Qdrant/Weaviate/Pinecone for true deletes.
        """
        col_docs = self._document_registry.get(collection_id, [])
        original_len = len(col_docs)
        self._document_registry[collection_id] = [
            d for d in col_docs if d["document_id"] != document_id
        ]
        if len(self._document_registry[collection_id]) < original_len:
            self._save_registry()
            return True
        return False

    def list_collections(self) -> List[dict]:
        index_dir = Path(self.settings.faiss_index_path)
        collections = []
        for col_id, docs in self._document_registry.items():
            collections.append(
                {
                    "id": col_id,
                    "name": col_id,
                    "document_count": len(docs),
                    "description": "",
                }
            )
        # Also include "default" even if empty
        if "default" not in self._document_registry:
            collections.insert(0, {"id": "default", "name": "default", "document_count": 0, "description": "Default collection"})
        return collections

    def get_stats(self) -> dict:
        total_docs = sum(len(v) for v in self._document_registry.values())
        return {
            "collections_count": len(self._document_registry),
            "documents_count": total_docs,
        }


# Singleton
_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service
