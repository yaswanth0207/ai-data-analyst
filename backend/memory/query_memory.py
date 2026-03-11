"""
Query memory: store and retrieve past questions/answers in ChromaDB for similar-question hints.
"""
import os
import uuid
from typing import Any

import chromadb
from chromadb.config import Settings


# Default persist directory from env
def _persist_dir() -> str:
    return os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")


# Singleton client and collection
_client: chromadb.PersistentClient | None = None
_collection_name = "query_memory"


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=_persist_dir(), settings=Settings(anonymized_telemetry=False))
    return _client


def _get_collection():
    return _get_client().get_or_create_collection(
        name=_collection_name,
        metadata={"description": "Past user questions and answers for the data analyst"},
    )


def add_query(question: str, answer: str, metadata: dict[str, Any] | None = None) -> str:
    """Store a question and its answer. Returns the document id."""
    coll = _get_collection()
    doc_id = str(uuid.uuid4())
    # Use question as document; store answer in metadata for retrieval
    meta = metadata or {}
    meta["answer"] = answer[:2000]  # Cap length for metadata
    coll.add(
        ids=[doc_id],
        documents=[question],
        metadatas=[meta],
    )
    return doc_id


def search_similar(question: str, n_results: int = 3) -> list[dict[str, Any]]:
    """
    Search for similar past questions. Returns list of dicts with 'question', 'answer', 'distance'.
    """
    coll = _get_collection()
    try:
        results = coll.query(
            query_texts=[question],
            n_results=min(n_results, 10),
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    if results and results["ids"] and len(results["ids"][0]) > 0:
        for i, doc_id in enumerate(results["ids"][0]):
            meta = (results["metadatas"] or [None])[0]
            meta = (meta or [{}])[i] if meta else {}
            # Only return if reasonably similar (e.g. distance < 1.5 for cosine)
            dist = (results.get("distances") or [[0]])[0][i] if results.get("distances") else 0
            out.append({
                "question": (results["documents"] or [[]])[0][i] if results.get("documents") else "",
                "answer": meta.get("answer", ""),
                "distance": float(dist) if dist is not None else 0,
            })
    return out
