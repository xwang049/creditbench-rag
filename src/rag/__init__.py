"""RAG module for creditbench data retrieval and generation.

Primary RAG implementation: Text-to-SQL RAG (sql_retriever.py)
Legacy implementations: Vector embeddings (disabled, see embeddings.py)
"""

# Note: EmbeddingService is disabled (see embeddings.py)
from .embeddings import EmbeddingService
from .retriever import VectorRetriever

# Text-to-SQL RAG (primary implementation)
from .sql_retriever import (
    sql_rag_answer,
    text_to_sql,
    execute_safe_sql,
    get_schema_description,
)

# Lazy import to avoid dependencies
def __getattr__(name):
    if name == "RAGChain":
        from .chain import RAGChain
        return RAGChain
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Text-to-SQL RAG (primary - use this!)
    "sql_rag_answer",
    "text_to_sql",
    "execute_safe_sql",
    "get_schema_description",
    # Legacy (disabled)
    "EmbeddingService",
    "VectorRetriever",
    "RAGChain",
]
