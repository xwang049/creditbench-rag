"""Vector and SQL-based retrieval for creditbench data."""

from typing import List, Dict, Any, Optional
from sqlalchemy import select, text, and_, or_
from sqlalchemy.orm import Session
from pgvector.sqlalchemy import Vector

from src.db.models import Company, CreditEvent, MacroIndicator
from src.rag.embeddings import EmbeddingService


class VectorRetriever:
    """Retriever that combines SQL queries with vector similarity search."""

    def __init__(self, session: Session, embedding_service: Optional[EmbeddingService] = None):
        """Initialize retriever.

        Args:
            session: SQLAlchemy session
            embedding_service: Service for generating embeddings
        """
        self.session = session
        self.embeddings = embedding_service or EmbeddingService()

    def search_companies(
        self,
        query: str,
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Company]:
        """Search for companies using vector similarity.

        Args:
            query: Search query text
            limit: Maximum number of results
            filters: Optional filters (e.g., {'sector': 'Technology'})

        Returns:
            List of matching companies
        """
        query_embedding = self.embeddings.embed_text(query)

        # Build base query
        stmt = select(Company)

        # Apply filters
        if filters:
            conditions = []
            for key, value in filters.items():
                if hasattr(Company, key):
                    conditions.append(getattr(Company, key) == value)
            if conditions:
                stmt = stmt.where(and_(*conditions))

        # Add vector similarity ordering
        if hasattr(Company, 'embedding'):
            stmt = stmt.order_by(
                Company.embedding.l2_distance(query_embedding)
            ).limit(limit)
        else:
            stmt = stmt.limit(limit)

        return list(self.session.execute(stmt).scalars().all())

    def search_credit_events(
        self,
        query: str,
        limit: int = 10,
        event_types: Optional[List[str]] = None,
        date_range: Optional[tuple] = None
    ) -> List[CreditEvent]:
        """Search for credit events using vector similarity.

        Args:
            query: Search query text
            limit: Maximum number of results
            event_types: Filter by event types
            date_range: Tuple of (start_date, end_date)

        Returns:
            List of matching credit events
        """
        query_embedding = self.embeddings.embed_text(query)

        # Build base query
        stmt = select(CreditEvent)

        # Apply filters
        conditions = []
        if event_types:
            conditions.append(CreditEvent.event_type.in_(event_types))
        if date_range:
            start_date, end_date = date_range
            if start_date:
                conditions.append(CreditEvent.event_date >= start_date)
            if end_date:
                conditions.append(CreditEvent.event_date <= end_date)

        if conditions:
            stmt = stmt.where(and_(*conditions))

        # Add vector similarity ordering
        if hasattr(CreditEvent, 'embedding'):
            stmt = stmt.order_by(
                CreditEvent.embedding.l2_distance(query_embedding)
            ).limit(limit)
        else:
            stmt = stmt.limit(limit)

        return list(self.session.execute(stmt).scalars().all())

    def get_company_context(self, company_id: int) -> Dict[str, Any]:
        """Get comprehensive context for a company.

        Args:
            company_id: Company ID

        Returns:
            Dictionary with company info, credit events, and related data
        """
        # Get company
        company = self.session.get(Company, company_id)
        if not company:
            return {}

        # Get recent credit events
        events_stmt = (
            select(CreditEvent)
            .where(CreditEvent.company_id == company_id)
            .order_by(CreditEvent.event_date.desc())
            .limit(10)
        )
        events = list(self.session.execute(events_stmt).scalars().all())

        return {
            'company': company,
            'credit_events': events,
            'event_count': len(events),
        }

    def hybrid_search(
        self,
        query: str,
        limit: int = 10,
        include_companies: bool = True,
        include_events: bool = True
    ) -> Dict[str, List[Any]]:
        """Perform hybrid search across multiple entity types.

        Args:
            query: Search query text
            limit: Maximum number of results per type
            include_companies: Whether to search companies
            include_events: Whether to search credit events

        Returns:
            Dictionary with search results by entity type
        """
        results = {}

        if include_companies:
            results['companies'] = self.search_companies(query, limit=limit)

        if include_events:
            results['credit_events'] = self.search_credit_events(query, limit=limit)

        return results

    def get_similar_companies(
        self,
        company_id: int,
        limit: int = 5
    ) -> List[Company]:
        """Find similar companies based on vector similarity.

        Args:
            company_id: Reference company ID
            limit: Maximum number of similar companies

        Returns:
            List of similar companies
        """
        company = self.session.get(Company, company_id)
        if not company or not hasattr(company, 'embedding') or company.embedding is None:
            return []

        stmt = (
            select(Company)
            .where(Company.id != company_id)
            .order_by(Company.embedding.l2_distance(company.embedding))
            .limit(limit)
        )

        return list(self.session.execute(stmt).scalars().all())
