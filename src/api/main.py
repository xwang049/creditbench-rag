"""FastAPI endpoints for creditbench RAG system."""

from typing import Optional, List
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.db.session import get_db
from src.rag.chain import RAGChain
from src.rag.retriever import VectorRetriever


app = FastAPI(
    title="CreditBench RAG API",
    description="RAG system for querying credit data with natural language",
    version="0.1.0"
)


# Request/Response Models
class QueryRequest(BaseModel):
    """Request model for general queries."""
    question: str = Field(..., description="Natural language question")
    max_results: int = Field(5, ge=1, le=20, description="Maximum results to retrieve")
    retrieve_companies: bool = Field(True, description="Include company data")
    retrieve_events: bool = Field(True, description="Include credit event data")


class CompanyQueryRequest(BaseModel):
    """Request model for company-specific queries."""
    company_id: int = Field(..., description="Company ID")
    question: str = Field(..., description="Natural language question about the company")


class SearchRequest(BaseModel):
    """Request model for vector search."""
    query: str = Field(..., description="Search query")
    limit: int = Field(5, ge=1, le=20, description="Maximum results")
    filters: Optional[dict] = Field(None, description="Optional filters")


class QueryResponse(BaseModel):
    """Response model for queries."""
    answer: str
    sources: dict
    context: str
    question: str


# Dependency to get RAG chain
def get_rag_chain(db: Session = Depends(get_db)) -> RAGChain:
    """Get RAG chain instance."""
    return RAGChain(session=db)


# Dependency to get retriever
def get_retriever(db: Session = Depends(get_db)) -> VectorRetriever:
    """Get vector retriever instance."""
    return VectorRetriever(session=db)


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "creditbench-rag"}


# Query endpoints
@app.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    rag_chain: RAGChain = Depends(get_rag_chain)
):
    """Answer a natural language question using RAG.

    Args:
        request: Query request with question and parameters

    Returns:
        Answer with sources and context
    """
    try:
        result = rag_chain.query(
            question=request.question,
            retrieve_companies=request.retrieve_companies,
            retrieve_events=request.retrieve_events,
            max_results=request.max_results
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/query/company", response_model=QueryResponse)
async def query_company(
    request: CompanyQueryRequest,
    rag_chain: RAGChain = Depends(get_rag_chain)
):
    """Answer a question about a specific company.

    Args:
        request: Company query request

    Returns:
        Answer with company context and sources
    """
    try:
        result = rag_chain.query_company(
            company_id=request.company_id,
            question=request.question
        )

        if "not found" in result.get('answer', '').lower():
            raise HTTPException(status_code=404, detail=f"Company {request.company_id} not found")

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/search/companies")
async def search_companies(
    request: SearchRequest,
    retriever: VectorRetriever = Depends(get_retriever)
):
    """Search for companies using vector similarity.

    Args:
        request: Search request

    Returns:
        List of matching companies
    """
    try:
        companies = retriever.search_companies(
            query=request.query,
            limit=request.limit,
            filters=request.filters
        )

        return {
            'results': [
                {
                    'id': c.id,
                    'name': c.name,
                    'ticker': c.ticker,
                    'sector': c.sector,
                    'industry': c.industry
                }
                for c in companies
            ],
            'count': len(companies)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/search/events")
async def search_credit_events(
    request: SearchRequest,
    retriever: VectorRetriever = Depends(get_retriever)
):
    """Search for credit events using vector similarity.

    Args:
        request: Search request

    Returns:
        List of matching credit events
    """
    try:
        events = retriever.search_credit_events(
            query=request.query,
            limit=request.limit
        )

        return {
            'results': [
                {
                    'id': e.id,
                    'event_type': e.event_type,
                    'event_date': str(e.event_date),
                    'company_id': e.company_id
                }
                for e in events
            ],
            'count': len(events)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/companies/{company_id}/similar")
async def get_similar_companies(
    company_id: int,
    limit: int = 5,
    retriever: VectorRetriever = Depends(get_retriever)
):
    """Find similar companies based on vector similarity.

    Args:
        company_id: Reference company ID
        limit: Maximum number of results

    Returns:
        List of similar companies
    """
    try:
        companies = retriever.get_similar_companies(
            company_id=company_id,
            limit=limit
        )

        if not companies:
            raise HTTPException(
                status_code=404,
                detail=f"Company {company_id} not found or has no embedding"
            )

        return {
            'reference_company_id': company_id,
            'similar_companies': [
                {
                    'id': c.id,
                    'name': c.name,
                    'ticker': c.ticker,
                    'sector': c.sector,
                    'industry': c.industry
                }
                for c in companies
            ],
            'count': len(companies)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
