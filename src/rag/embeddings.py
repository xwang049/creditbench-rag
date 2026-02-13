"""Embedding generation service for text and structured data.

⚠️  IMPORTANT: This module is currently DISABLED ⚠️

We are using Text-to-SQL RAG (see sql_retriever.py) instead of vector embeddings.
This file is kept for reference but should NOT be used in production.

If you need to re-enable vector embeddings in the future:
1. Uncomment the CreditEventEmbedding model in src/db/models.py
2. Uncomment the pgvector import in src/db/models.py
3. Re-enable the embedding columns on Company and CreditEvent models
4. Run database migrations to add back the tables/columns
5. Use the functions in this file to generate embeddings

For now, use src/rag/sql_retriever.py for all RAG functionality.
"""

from typing import List, Optional
import numpy as np
import logging
from tqdm import tqdm
from sqlalchemy.orm import Session
from sqlalchemy import text

try:
    from langchain_anthropic import AnthropicEmbeddings
    USE_ANTHROPIC = True
except ImportError:
    USE_ANTHROPIC = False

try:
    from sentence_transformers import SentenceTransformer
    USE_SENTENCE_TRANSFORMERS = True
except ImportError:
    USE_SENTENCE_TRANSFORMERS = False

from src.config import settings
from src.db.models import CreditEvent, Company, IndustryMapping
# Note: CreditEventEmbedding model is currently disabled (see sql_retriever.py for Text-to-SQL RAG)

logger = logging.getLogger(__name__)


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


def format_credit_event_text(event, company, industry_sector: Optional[str] = None) -> str:
    """Format credit event data into text for embedding.

    Args:
        event: CreditEvent object
        company: Company object
        industry_sector: Industry sector name (optional)

    Returns:
        Formatted text string
    """
    parts = []

    # Company name
    if company and company.company_name:
        parts.append(company.company_name)

    # Action name
    if event.action_name:
        parts.append(event.action_name)

    # Subcategory
    if event.subcategory:
        subcategory = event.subcategory[:200]  # Truncate if too long
        parts.append(subcategory)

    # Date
    if event.announcement_date:
        parts.append(f"Date: {event.announcement_date}")

    # Industry sector
    if industry_sector:
        parts.append(f"Sector: {industry_sector}")

    # Country
    if company and company.country_name:
        parts.append(f"Country: {company.country_name}")

    return " | ".join(parts)


def generate_credit_event_embeddings(
    session: Session,
    batch_size: int = 100,
    use_anthropic: bool = None,
    limit: Optional[int] = None
) -> dict:
    """⚠️  DISABLED: Generate embeddings for all credit events and store in database.

    This function is currently disabled because we're using Text-to-SQL RAG instead.
    See src/rag/sql_retriever.py for the active RAG implementation.

    Args:
        session: Database session
        batch_size: Number of events to process per batch
        use_anthropic: Whether to use Anthropic embeddings (default: auto-detect)
        limit: Maximum number of events to process (for testing)

    Returns:
        Dictionary with statistics
    """
    logger.warning("⚠️  generate_credit_event_embeddings() is DISABLED - using Text-to-SQL RAG instead (see sql_retriever.py)")
    logger.warning("This function will not work because CreditEventEmbedding model is disabled in models.py")
    return {'processed': 0, 'skipped': 0, 'total': 0, 'error': 'Embedding generation disabled - using Text-to-SQL RAG'}

    # Original code below is kept for reference but will not execute
    logger.info("Starting credit event embedding generation...")

    # Determine which embedding model to use
    if use_anthropic is None:
        use_anthropic = USE_ANTHROPIC and hasattr(settings, 'ANTHROPIC_API_KEY') and settings.ANTHROPIC_API_KEY

    if use_anthropic:
        logger.info("Using Anthropic voyage-finance-2 embeddings (1024 dimensions)")
        embedder = AnthropicEmbeddings(
            model="voyage-finance-2",
            anthropic_api_key=settings.ANTHROPIC_API_KEY
        )
        model_name = "voyage-finance-2"
    elif USE_SENTENCE_TRANSFORMERS:
        logger.info("Using sentence-transformers BAAI/bge-large-en-v1.5 (1024 dimensions)")
        embedder = SentenceTransformer('BAAI/bge-large-en-v1.5')
        model_name = "BAAI/bge-large-en-v1.5"
    else:
        raise RuntimeError("No embedding model available. Install sentence-transformers or provide Anthropic API key.")

    # Get industry mapping for context
    industry_map = {}
    industries = session.query(IndustryMapping).all()
    for ind in industries:
        if ind.industry_sector_num:
            industry_map[ind.industry_sector_num] = ind.industry_sector

    # Query credit events that don't have embeddings yet
    query = text("""
        SELECT ce.id
        FROM credit_events ce
        LEFT JOIN credit_event_embeddings cee ON ce.id = cee.credit_event_id
        WHERE cee.id IS NULL
        ORDER BY ce.id
    """)

    if limit:
        query = text(f"{query.text} LIMIT {limit}")

    event_ids = [row[0] for row in session.execute(query).fetchall()]
    total_events = len(event_ids)

    if total_events == 0:
        logger.info("All credit events already have embeddings")
        return {'processed': 0, 'skipped': 0, 'total': 0}

    logger.info(f"Found {total_events} credit events without embeddings")

    processed = 0
    skipped = 0

    # Process in batches
    for i in tqdm(range(0, total_events, batch_size), desc="Generating embeddings"):
        batch_ids = event_ids[i:i + batch_size]

        # Fetch batch of events with company info
        events = session.query(CreditEvent).filter(CreditEvent.id.in_(batch_ids)).all()
        companies = {
            c.u3_company_number: c
            for c in session.query(Company).filter(
                Company.u3_company_number.in_([e.u3_company_number for e in events])
            ).all()
        }

        # Format text for each event
        texts = []
        event_records = []

        for event in events:
            company = companies.get(event.u3_company_number)
            industry_sector = None
            if company and company.industry_sector_num:
                industry_sector = industry_map.get(company.industry_sector_num)

            event_text = format_credit_event_text(event, company, industry_sector)

            if not event_text or len(event_text) < 10:  # Skip events with insufficient data
                skipped += 1
                continue

            texts.append(event_text)
            event_records.append((event.id, event_text))

        if not texts:
            continue

        # Generate embeddings
        try:
            if use_anthropic:
                embeddings = embedder.embed_documents(texts)
            else:
                embeddings = embedder.encode(texts, convert_to_numpy=True).tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings for batch: {e}")
            skipped += len(texts)
            continue

        # Store in database
        for (event_id, event_text), embedding in zip(event_records, embeddings):
            try:
                emb_record = CreditEventEmbedding(
                    credit_event_id=event_id,
                    embedding=embedding,
                    text_content=event_text,
                    embedding_model=model_name
                )
                session.add(emb_record)
                processed += 1
            except Exception as e:
                logger.warning(f"Error storing embedding for event {event_id}: {e}")
                skipped += 1

        # Commit batch
        try:
            session.commit()
        except Exception as e:
            logger.error(f"Error committing batch: {e}")
            session.rollback()

    logger.info(f"Embedding generation complete: {processed} processed, {skipped} skipped")

    return {
        'processed': processed,
        'skipped': skipped,
        'total': total_events,
        'model': model_name
    }
