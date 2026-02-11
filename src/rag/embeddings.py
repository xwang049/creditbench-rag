"""Embedding generation service for text and structured data."""

from typing import List, Optional
import numpy as np
from langchain_anthropic import AnthropicEmbeddings
from src.config import settings


class EmbeddingService:
    """Service for generating embeddings using Anthropic's embedding models."""

    def __init__(self, model: Optional[str] = None):
        """Initialize embedding service.

        Args:
            model: Embedding model name. Defaults to settings.EMBEDDING_MODEL
        """
        self.model = model or settings.EMBEDDING_MODEL
        self.embeddings = AnthropicEmbeddings(
            model=self.model,
            anthropic_api_key=settings.ANTHROPIC_API_KEY
        )

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            List of embedding values
        """
        return self.embeddings.embed_query(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        return self.embeddings.embed_documents(texts)

    def embed_company(self, company_data: dict) -> List[float]:
        """Generate embedding for company data.

        Args:
            company_data: Dictionary containing company information

        Returns:
            Embedding vector
        """
        # Construct a rich text representation of the company
        text_parts = [
            f"Company: {company_data.get('name', '')}",
            f"Ticker: {company_data.get('ticker', '')}",
            f"Industry: {company_data.get('industry', '')}",
            f"Sector: {company_data.get('sector', '')}",
            f"Description: {company_data.get('description', '')}",
        ]

        # Add financial metrics if available
        if 'revenue' in company_data:
            text_parts.append(f"Revenue: {company_data['revenue']}")
        if 'market_cap' in company_data:
            text_parts.append(f"Market Cap: {company_data['market_cap']}")

        text = " | ".join(filter(None, text_parts))
        return self.embed_text(text)

    def embed_credit_event(self, event_data: dict) -> List[float]:
        """Generate embedding for credit event data.

        Args:
            event_data: Dictionary containing credit event information

        Returns:
            Embedding vector
        """
        text_parts = [
            f"Event Type: {event_data.get('event_type', '')}",
            f"Date: {event_data.get('event_date', '')}",
            f"Company: {event_data.get('company_name', '')}",
            f"Rating: {event_data.get('rating', '')}",
            f"Description: {event_data.get('description', '')}",
        ]

        text = " | ".join(filter(None, text_parts))
        return self.embed_text(text)

    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score
        """
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
