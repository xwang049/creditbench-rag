# CreditBench RAG

A Retrieval-Augmented Generation (RAG) system for analyzing credit data using PostgreSQL with pgvector, LangChain, and Anthropic's Claude.

## Overview

This project provides a RAG-based question-answering system over credit benchmark data, including:
- Company information and profiles
- Credit events (defaults, bankruptcies, downgrades, etc.)
- Macroeconomic indicators
- Vector similarity search for semantic retrieval
- Natural language querying with Claude

## Architecture

```
creditbench-rag/
├── src/
│   ├── config.py           # Configuration and environment variables
│   ├── db/                 # Database layer
│   │   ├── models.py       # SQLAlchemy ORM models
│   │   ├── session.py      # Database session management
│   │   └── init_db.py      # Database initialization
│   ├── ingestion/          # Data loading pipeline
│   │   ├── load_companies.py
│   │   ├── load_credit_events.py
│   │   ├── load_macros.py
│   │   └── load_all.py
│   ├── rag/                # RAG components
│   │   ├── embeddings.py   # Embedding generation
│   │   ├── retriever.py    # Vector similarity search
│   │   └── chain.py        # RAG chain with LLM
│   └── api/                # FastAPI endpoints
│       └── main.py
├── scripts/
│   └── seed.py             # One-click database seeding
├── tests/
│   └── test_queries.py     # Test suite
├── docker-compose.yml      # PostgreSQL + pgvector
├── pyproject.toml          # Python dependencies
└── .env.example            # Environment variables template
```

## Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Anthropic API key

## Quick Start

### 1. Clone and Setup Environment

```bash
# Clone the repository
cd creditbench-rag

# Copy environment template
cp .env.example .env

# Edit .env and add your Anthropic API key
# ANTHROPIC_API_KEY=your-key-here
```

### 2. Start PostgreSQL with pgvector

```bash
docker-compose up -d
```

This starts a PostgreSQL 15 container with the pgvector extension enabled.

### 3. Install Python Dependencies

Using `uv` (recommended):
```bash
uv pip install -e .
```

Or using `pip`:
```bash
pip install -e .
```

For development dependencies:
```bash
uv pip install -e ".[dev]"
```

### 4. Load Data

```bash
python scripts/seed.py
```

This will:
- Initialize the database and create tables
- Load company data
- Load credit events
- Load macroeconomic indicators
- Generate embeddings for vector search

### 5. Run Queries

#### Using Python API

```python
from src.db.session import get_session
from src.rag.chain import RAGChain

# Create RAG chain
with get_session() as session:
    chain = RAGChain(session)

    # Ask a question
    result = chain.query(
        question="What are recent defaults in the technology sector?",
        max_results=5
    )

    print(result['answer'])
    print("\nSources:", result['sources'])
```

#### Using REST API

Start the API server:
```bash
python -m src.api.main
# or
uvicorn src.api.main:app --reload
```

Then query via HTTP:
```bash
# General query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are recent defaults in the technology sector?",
    "max_results": 5
  }'

# Company-specific query
curl -X POST http://localhost:8000/query/company \
  -H "Content-Type: application/json" \
  -d '{
    "company_id": 1,
    "question": "What is the credit history of this company?"
  }'

# Vector search for companies
curl -X POST http://localhost:8000/search/companies \
  -H "Content-Type: application/json" \
  -d '{
    "query": "technology companies with high revenue",
    "limit": 5
  }'
```

API documentation available at: http://localhost:8000/docs

## Features

### Vector Similarity Search

The system uses pgvector for efficient similarity search:
- Company embeddings based on name, sector, industry, and description
- Credit event embeddings based on event type, date, and description
- Hybrid search combining SQL filters with vector similarity

### RAG Chain

The RAG chain combines:
1. **Retrieval**: Vector similarity search to find relevant companies and events
2. **Context Formation**: Structure retrieved data into formatted context
3. **Generation**: Use Claude to generate natural language answers

### Supported Queries

- "What are recent credit events in the energy sector?"
- "Which companies have had downgrades in the last year?"
- "Show me technology companies with high credit risk"
- "What is the credit profile of [company name]?"
- "Find similar companies to [company name]"

## Database Schema

### Companies Table
- Basic info: name, ticker, sector, industry
- Financial metrics: revenue, market cap, etc.
- Vector embedding for similarity search

### Credit Events Table
- Event details: type, date, rating
- Company reference (foreign key)
- Vector embedding for semantic search

### Macro Indicators Table
- Economic indicators over time
- Used for context in credit analysis

## Development

### Running Tests

```bash
pytest tests/
```

For integration tests (requires database):
```bash
pytest tests/ -m integration
```

### Code Formatting

```bash
black src/ tests/
ruff check src/ tests/
```

### Adding New Data Sources

1. Create a loader in `src/ingestion/`
2. Implement data parsing and validation
3. Add embedding generation
4. Update `load_all.py` to include new loader
5. Add tests in `tests/`

## Configuration

Key environment variables in `.env`:

```bash
# Database
DATABASE_URL=postgresql://creditbench:creditbench@localhost:5432/creditbench

# API Keys
ANTHROPIC_API_KEY=your-key-here

# Model Configuration
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=claude-3-5-sonnet-20241022

# Application
LOG_LEVEL=INFO
```

## Performance Tips

1. **Batch Embedding Generation**: Generate embeddings in batches for efficiency
2. **Index Optimization**: Ensure pgvector indexes are created (see `init-db.sql`)
3. **Connection Pooling**: Use SQLAlchemy connection pooling for concurrent requests
4. **Caching**: Consider caching frequent queries at the application layer

## Troubleshooting

### Database Connection Issues
```bash
# Check if PostgreSQL is running
docker-compose ps

# View logs
docker-compose logs postgres

# Restart container
docker-compose restart postgres
```

### Embedding Generation Errors
- Verify `ANTHROPIC_API_KEY` is set correctly
- Check API rate limits
- Ensure model name is valid

### Slow Queries
- Check if pgvector indexes exist: `\d+ companies` in psql
- Reduce `max_results` parameter
- Consider adding more specific filters

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run tests and formatting checks
5. Submit a pull request

## License

[Your License Here]

## References

- [pgvector](https://github.com/pgvector/pgvector) - Vector similarity search for PostgreSQL
- [LangChain](https://python.langchain.com/) - Framework for LLM applications
- [Anthropic Claude](https://www.anthropic.com/claude) - AI assistant for generation
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework for APIs
