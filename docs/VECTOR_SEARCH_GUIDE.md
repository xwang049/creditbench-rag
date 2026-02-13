# Vector Search Guide

## Overview

CreditBench RAG now supports **hybrid retrieval** combining:
1. **Vector Search**: Semantic similarity using pgvector + embeddings
2. **SQL Generation**: Natural language to SQL using Claude
3. **Hybrid Search**: Combines both methods for comprehensive results

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Query                              │
└─────────────┬────────────────────────────────┬──────────────┘
              │                                │
       ┌──────▼──────┐                  ┌─────▼────────┐
       │ Vector Search│                  │ SQL Generate │
       │ (Semantic)   │                  │ (Structured) │
       └──────┬───────┘                  └─────┬────────┘
              │                                │
       ┌──────▼──────────────┐          ┌─────▼──────────────┐
       │ Query → Embedding   │          │ Query → SQL        │
       │ Cosine Similarity   │          │ Execute            │
       │ Top-K Results       │          │ Parse Results      │
       └──────┬──────────────┘          └─────┬──────────────┘
              │                                │
              └────────────┬───────────────────┘
                           │
                    ┌──────▼─────────┐
                    │  Merge & Dedup │
                    │  Rank Results  │
                    └──────┬─────────┘
                           │
                    ┌──────▼─────────┐
                    │  Final Results │
                    └────────────────┘
```

## Setup

### 1. Install Dependencies

```bash
# For Anthropic embeddings (recommended)
pip install anthropic langchain-anthropic

# OR for local embeddings (free, no API key)
pip install sentence-transformers torch
```

### 2. Create Tables and Generate Embeddings

```bash
# Run the setup script
python -m scripts.setup_vector_search
```

This will:
- Create `credit_event_embeddings` table
- Generate embeddings for all ~41K credit events
- Create HNSW index for fast similarity search
- Test retrieval functions

**Embedding Models:**
- **Anthropic voyage-finance-2**: 1024 dims, optimized for finance (requires API key)
- **BAAI/bge-large-en-v1.5**: 1024 dims, local, free (no API key needed)

**Time Estimate:**
- With Anthropic API: ~5-10 minutes (41K events / 100 per batch)
- With local model: ~15-30 minutes (first run downloads model)

## Usage

### Vector Retrieve (Semantic Search)

Find semantically similar credit events:

```python
from src.db.session import get_session
from src.rag.retriever import vector_retrieve

query = "technology companies filing for bankruptcy"

with get_session() as session:
    results = vector_retrieve(query, session, top_k=10)

    for r in results:
        print(f"{r['company_name']} - {r['action_name']}")
        print(f"  Date: {r['announcement_date']}")
        print(f"  Score: {r['similarity_score']:.4f}")
        print(f"  Sector: {r['industry_sector']}")
        print()
```

**When to use:**
- Fuzzy/semantic queries
- Finding similar events
- Exploring related companies
- When exact SQL is unclear

### SQL Retrieve (Structured Query)

Convert natural language to SQL:

```python
from src.db.session import get_session
from src.rag.retriever import sql_retrieve

query = "Show me the top 10 companies by number of credit events in 2023"

with get_session() as session:
    results = sql_retrieve(query, session, max_results=10)

    for r in results:
        print(r)
```

**When to use:**
- Precise queries (dates, counts, aggregations)
- Complex filters and joins
- Statistical analysis
- Structured reporting

**Security:**
- Only SELECT statements allowed
- Dangerous keywords (INSERT, DELETE, DROP) are blocked
- All queries are logged

### Hybrid Retrieve (Best of Both)

Combines vector + SQL for comprehensive results:

```python
from src.db.session import get_session
from src.rag.retriever import hybrid_retrieve

query = "financial distress in retail companies"

with get_session() as session:
    results = hybrid_retrieve(query, session, top_k=10)

    for r in results:
        method = r.get('retrieval_method', 'unknown')
        print(f"[{method}] {r.get('company_name')} - {r.get('action_name')}")
```

**When to use:**
- General questions
- When unsure which method is better
- Maximum recall
- Production RAG systems

### Using the VectorRetriever Class

```python
from src.db.session import get_session
from src.rag.retriever import VectorRetriever

with get_session() as session:
    retriever = VectorRetriever(session)

    # Choose method: 'sql', 'vector', or 'hybrid'
    results = retriever.search(
        "bankruptcy filings in tech sector",
        method="hybrid",
        top_k=10
    )

    # Get credit events for specific company
    events = retriever.get_company_credit_events(
        u3_company_number=26978,
        limit=10
    )
```

## Database Schema

### credit_event_embeddings

```sql
CREATE TABLE credit_event_embeddings (
    id SERIAL PRIMARY KEY,
    credit_event_id INTEGER UNIQUE REFERENCES credit_events(id),
    embedding vector(1024) NOT NULL,
    text_content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    embedding_model VARCHAR(100)
);

-- HNSW index for fast cosine similarity search
CREATE INDEX idx_credit_event_embeddings_hnsw
ON credit_event_embeddings
USING hnsw (embedding vector_cosine_ops);
```

### Text Format for Embeddings

Each credit event is converted to text:

```
{company_name} | {action_name} | {subcategory} | Date: {announcement_date} | Sector: {industry_sector} | Country: {country_name}
```

Example:
```
Tesla Inc | Delisting | Moved to OTC market | Date: 2023-05-15 | Sector: Consumer Discretionary | Country: United States
```

## Vector Search Queries

### Cosine Similarity

pgvector uses `<=>` operator for cosine distance:

```sql
SELECT *
FROM credit_event_embeddings
ORDER BY embedding <=> '[query_embedding]'
LIMIT 10;
```

**Distance metrics:**
- `<->`: L2 distance (Euclidean)
- `<#>`: Inner product
- `<=>`: Cosine distance (1 - cosine similarity)

**Similarity score:**
```
similarity_score = 1 - cosine_distance
```

Higher score = more similar (0 to 1)

## Performance

### Index Types

| Index Type | Build Time | Query Speed | Recall |
|------------|-----------|-------------|--------|
| IVFFlat    | Fast      | Good        | ~95%   |
| HNSW       | Slower    | Excellent   | ~98%   |

**Current setup:** HNSW (Hierarchical Navigable Small World)
- Best query performance
- Slightly longer build time
- Higher accuracy

### Query Performance

- **Vector search**: ~10-50ms for top-10
- **SQL generation**: ~500-1000ms (LLM call)
- **Hybrid search**: ~1-2 seconds

### Optimization Tips

1. **Use appropriate top_k**: Don't fetch more than needed
2. **Add filters**: Pre-filter by date/sector before vector search
3. **Batch queries**: Process multiple queries together
4. **Cache embeddings**: Store user query embeddings for repeated searches

## Example Queries

### 1. Find Similar Events

```python
query = "major bankruptcy filing with high debt"
results = vector_retrieve(query, session, top_k=5)
```

### 2. Time-based Analysis

```python
query = "Count credit events by month for tech companies in 2023"
results = sql_retrieve(query, session)
```

### 3. Sector Comparison

```python
query = "bankruptcy rates in energy vs financial sectors"
results = hybrid_retrieve(query, session, top_k=20)
```

### 4. Company Research

```python
query = "Companies with debt restructuring in last 6 months"
results = vector_retrieve(query, session, top_k=15)
```

## Troubleshooting

### No results from vector_retrieve

**Cause:** Embeddings not generated

**Solution:**
```bash
python -m scripts.setup_vector_search
```

### "No embedding model available"

**Cause:** Neither Anthropic API key nor sentence-transformers installed

**Solution:**
```bash
# Option 1: Add API key to .env
ANTHROPIC_API_KEY=sk-ant-...

# Option 2: Install sentence-transformers
pip install sentence-transformers torch
```

### SQL generation fails

**Cause:** No Anthropic API key

**Solution:**
```bash
# Add to .env file
ANTHROPIC_API_KEY=sk-ant-...
```

### Slow vector search

**Cause:** Missing HNSW index

**Solution:**
```sql
-- Manually create index
CREATE INDEX idx_credit_event_embeddings_hnsw
ON credit_event_embeddings
USING hnsw (embedding vector_cosine_ops);
```

### Out of memory during embedding generation

**Cause:** Too large batch size

**Solution:**
Reduce batch size in `generate_credit_event_embeddings`:
```python
stats = generate_credit_event_embeddings(
    session,
    batch_size=50,  # Reduce from 100
    limit=1000      # Process in chunks
)
```

## Advanced: Custom Retrieval

### Add Filters to Vector Search

```python
from sqlalchemy import text

query_embedding = embedder.embed_query("bankruptcy")

sql = text("""
    SELECT *
    FROM credit_event_embeddings cee
    JOIN credit_events ce ON cee.credit_event_id = ce.id
    JOIN companies c ON ce.u3_company_number = c.u3_company_number
    WHERE c.country_name = 'United States'
      AND ce.announcement_date >= '2023-01-01'
    ORDER BY cee.embedding <=> :embedding
    LIMIT 10
""")

results = session.execute(sql, {"embedding": query_embedding})
```

### Rerank Results

```python
from src.rag.embeddings import EmbeddingService

service = EmbeddingService()
query_embedding = service.embed_text(query)

# Get candidates
candidates = vector_retrieve(query, session, top_k=50)

# Rerank by computing exact similarity
for candidate in candidates:
    candidate_embedding = candidate['embedding']  # If stored
    score = service.cosine_similarity(query_embedding, candidate_embedding)
    candidate['rerank_score'] = score

# Sort by rerank score
candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
top_results = candidates[:10]
```

## API Integration

See `src/api/main.py` for FastAPI endpoints:

```python
@app.post("/search")
async def search(query: str, method: str = "hybrid", top_k: int = 10):
    with get_session() as session:
        retriever = VectorRetriever(session)
        results = retriever.search(query, method=method, top_k=top_k)
        return {"results": results}
```

## Next Steps

1. ✅ Setup vector search
2. ✅ Test retrieval functions
3. 🔲 Integrate with RAG chain (see `src/rag/chain.py`)
4. 🔲 Add reranking
5. 🔲 Implement query expansion
6. 🔲 Add retrieval evaluation metrics

---

**Last Updated**: 2026-02-12
**pgvector Version**: 0.5.0+
**Embedding Dimensions**: 1024
