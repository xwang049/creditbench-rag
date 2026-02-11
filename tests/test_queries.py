"""Tests for RAG query functionality."""

import pytest
from sqlalchemy.orm import Session

from src.db.session import get_session
from src.rag.chain import RAGChain
from src.rag.retriever import VectorRetriever
from src.rag.embeddings import EmbeddingService


@pytest.fixture
def db_session():
    """Provide a database session for tests."""
    with get_session() as session:
        yield session


@pytest.fixture
def embedding_service():
    """Provide an embedding service instance."""
    return EmbeddingService()


@pytest.fixture
def retriever(db_session):
    """Provide a retriever instance."""
    return VectorRetriever(db_session)


@pytest.fixture
def rag_chain(db_session):
    """Provide a RAG chain instance."""
    return RAGChain(db_session)


class TestEmbeddingService:
    """Tests for embedding generation."""

    def test_embed_text(self, embedding_service):
        """Test single text embedding."""
        text = "Apple Inc. is a technology company"
        embedding = embedding_service.embed_text(text)

        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)

    def test_embed_texts(self, embedding_service):
        """Test batch text embedding."""
        texts = [
            "Apple Inc. is a technology company",
            "Microsoft Corporation develops software"
        ]
        embeddings = embedding_service.embed_texts(texts)

        assert len(embeddings) == 2
        assert all(isinstance(emb, list) for emb in embeddings)

    def test_embed_company(self, embedding_service):
        """Test company data embedding."""
        company_data = {
            'name': 'Apple Inc.',
            'ticker': 'AAPL',
            'industry': 'Technology',
            'sector': 'Consumer Electronics',
            'description': 'Designs and manufactures consumer electronics'
        }

        embedding = embedding_service.embed_company(company_data)
        assert isinstance(embedding, list)
        assert len(embedding) > 0

    def test_cosine_similarity(self, embedding_service):
        """Test cosine similarity calculation."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        vec3 = [1.0, 0.0, 0.0]

        # Orthogonal vectors
        sim1 = embedding_service.cosine_similarity(vec1, vec2)
        assert abs(sim1) < 0.01

        # Identical vectors
        sim2 = embedding_service.cosine_similarity(vec1, vec3)
        assert abs(sim2 - 1.0) < 0.01


class TestVectorRetriever:
    """Tests for vector retrieval."""

    def test_search_companies(self, retriever):
        """Test company search."""
        results = retriever.search_companies(
            query="technology companies",
            limit=5
        )

        assert isinstance(results, list)
        # Note: May be empty if database is not seeded

    def test_search_companies_with_filters(self, retriever):
        """Test company search with filters."""
        results = retriever.search_companies(
            query="companies",
            limit=5,
            filters={'sector': 'Technology'}
        )

        assert isinstance(results, list)

    def test_search_credit_events(self, retriever):
        """Test credit event search."""
        results = retriever.search_credit_events(
            query="bankruptcy",
            limit=10
        )

        assert isinstance(results, list)

    def test_hybrid_search(self, retriever):
        """Test hybrid search across multiple types."""
        results = retriever.hybrid_search(
            query="technology defaults",
            limit=5
        )

        assert isinstance(results, dict)
        assert 'companies' in results
        assert 'credit_events' in results


class TestRAGChain:
    """Tests for RAG chain."""

    def test_query_basic(self, rag_chain):
        """Test basic query."""
        result = rag_chain.query(
            question="What are some recent credit events?",
            max_results=5
        )

        assert isinstance(result, dict)
        assert 'answer' in result
        assert 'sources' in result
        assert 'context' in result
        assert 'question' in result
        assert isinstance(result['answer'], str)

    def test_query_company_specific(self, rag_chain, db_session):
        """Test company-specific query."""
        # This test requires at least one company in the database
        from src.db.models import Company

        company = db_session.query(Company).first()
        if not company:
            pytest.skip("No companies in database")

        result = rag_chain.query_company(
            company_id=company.id,
            question="What is the credit profile of this company?"
        )

        assert isinstance(result, dict)
        assert 'answer' in result
        assert 'company' in result
        assert result['company']['id'] == company.id

    def test_query_with_filters(self, rag_chain):
        """Test query with specific filters."""
        result = rag_chain.query(
            question="Which technology companies have had defaults?",
            retrieve_companies=True,
            retrieve_events=True,
            max_results=3
        )

        assert isinstance(result, dict)
        assert len(result['sources']['companies']) <= 3
        assert len(result['sources']['credit_events']) <= 3


@pytest.mark.integration
class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_rag_pipeline(self, db_session):
        """Test complete RAG pipeline from query to answer."""
        # Create RAG chain
        chain = RAGChain(db_session)

        # Run query
        result = chain.query(
            question="What are the main credit risks in the technology sector?",
            max_results=5
        )

        # Verify response structure
        assert result['answer']
        assert isinstance(result['sources'], dict)
        assert result['question'] == "What are the main credit risks in the technology sector?"

    def test_company_analysis_workflow(self, db_session):
        """Test company analysis workflow."""
        from src.db.models import Company

        # Get a company
        company = db_session.query(Company).first()
        if not company:
            pytest.skip("No companies in database")

        # Initialize components
        retriever = VectorRetriever(db_session)
        chain = RAGChain(db_session)

        # Get company context
        context = retriever.get_company_context(company.id)
        assert 'company' in context
        assert 'credit_events' in context

        # Query about company
        result = chain.query_company(
            company_id=company.id,
            question="Summarize the credit history"
        )

        assert result['answer']
        assert result['company']['id'] == company.id
