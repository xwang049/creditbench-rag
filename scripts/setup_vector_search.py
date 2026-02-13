"""Setup vector search for credit events.

This script:
1. Creates the credit_event_embeddings table
2. Generates embeddings for all credit events
3. Creates HNSW index for fast similarity search
4. Tests vector and hybrid retrieval
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import sys
from sqlalchemy import text

# Direct imports to avoid __init__.py chain import issues
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.db.session import get_session, engine
from src.db.models import Base, CreditEventEmbedding

# Import functions directly from modules
import src.rag.embeddings as embeddings_module
import src.rag.retriever as retriever_module

generate_credit_event_embeddings = embeddings_module.generate_credit_event_embeddings
vector_retrieve = retriever_module.vector_retrieve
sql_retrieve = retriever_module.sql_retrieve
hybrid_retrieve = retriever_module.hybrid_retrieve


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def create_tables():
    """Create credit_event_embeddings table if it doesn't exist."""
    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("Creating credit_event_embeddings table...")
    logger.info("=" * 70)

    try:
        # Only create tables that don't exist yet
        Base.metadata.create_all(bind=engine, checkfirst=True)
        logger.info("[OK] credit_event_embeddings table created (or already exists)")
        return True
    except Exception as e:
        logger.error(f"[ERROR] Failed to create table: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_setup():
    """Verify the vector search setup."""
    logger = logging.getLogger(__name__)

    logger.info("\n" + "=" * 70)
    logger.info("Verifying vector search setup...")
    logger.info("=" * 70)

    with get_session() as session:
        # Check table exists
        result = session.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'credit_event_embeddings'
            )
        """)).scalar()

        if not result:
            logger.error("[FAIL] credit_event_embeddings table not found")
            return False

        logger.info("[OK] credit_event_embeddings table exists")

        # Check embedding count
        count = session.query(CreditEventEmbedding).count()
        logger.info(f"[OK] Found {count:,} embeddings in database")

        # Check index exists
        result = session.execute(text("""
            SELECT EXISTS (
                SELECT FROM pg_indexes
                WHERE tablename = 'credit_event_embeddings'
                  AND indexname LIKE '%embedding%'
            )
        """)).scalar()

        if result:
            logger.info("[OK] Vector index exists")
        else:
            logger.warning("[WARN] Vector index not found (will be slow)")

        # Sample embedding
        sample = session.query(CreditEventEmbedding).first()
        if sample:
            logger.info(f"[OK] Sample embedding: {len(sample.embedding)} dimensions")
            logger.info(f"     Model: {sample.embedding_model}")
            logger.info(f"     Text: {sample.text_content[:100]}...")

        return True


def test_retrieval():
    """Test retrieval functions."""
    logger = logging.getLogger(__name__)

    logger.info("\n" + "=" * 70)
    logger.info("Testing retrieval functions...")
    logger.info("=" * 70)

    test_queries = [
        "bankruptcy filings in technology sector",
        "companies defaulting in 2023",
        "delisting events"
    ]

    with get_session() as session:
        for query in test_queries:
            logger.info(f"\nQuery: '{query}'")
            logger.info("-" * 70)

            # Test vector retrieve
            try:
                results = vector_retrieve(query, session, top_k=3)
                logger.info(f"[Vector] Found {len(results)} results")
                if results:
                    for i, r in enumerate(results[:2], 1):
                        logger.info(f"  {i}. {r.get('company_name', 'N/A')} - {r.get('action_name', 'N/A')}")
                        logger.info(f"     Score: {r.get('similarity_score', 0):.4f}")
            except Exception as e:
                logger.warning(f"[Vector] Error: {e}")

            # Test hybrid retrieve
            try:
                results = hybrid_retrieve(query, session, top_k=3)
                logger.info(f"[Hybrid] Found {len(results)} results")
            except Exception as e:
                logger.warning(f"[Hybrid] Error: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Setup vector search for credit events')
    parser.add_argument('--auto-confirm', '-y', action='store_true',
                       help='Automatically confirm all prompts')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of events to process (for testing)')
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    print()
    print("=" * 70)
    print("  Setup Vector Search for CreditBench")
    print("=" * 70)
    print()

    # Step 1: Create table
    print("Step 1: Creating credit_event_embeddings table...")
    print("-" * 70)
    if not create_tables():
        return 1
    print()

    # Step 2: Check if embeddings already exist
    with get_session() as session:
        existing_count = session.query(CreditEventEmbedding).count()
        total_events = session.execute(text("SELECT COUNT(*) FROM credit_events")).scalar()

    if existing_count > 0:
        logger.info(f"Found {existing_count:,} existing embeddings")
        if args.auto_confirm:
            response = 'y'
            logger.info("Auto-confirm: Regenerating embeddings")
        else:
            response = input(f"\nRegenerate all embeddings? [y/N]: ").strip().lower()

        if response != 'y':
            logger.info("Skipping embedding generation")
            skip_generation = True
        else:
            skip_generation = False
            logger.info("Clearing existing embeddings...")
            with get_session() as session:
                session.query(CreditEventEmbedding).delete()
                session.commit()
    else:
        skip_generation = False

    # Step 3: Generate embeddings
    if not skip_generation:
        print("Step 2: Generating embeddings for credit events...")
        print("-" * 70)
        print(f"Total credit events: {total_events:,}")
        print()
        print("NOTE: This will take a while. Embeddings are generated in batches of 100.")
        print("      You can use Ctrl+C to stop and resume later (idempotent).")
        print()

        if args.auto_confirm:
            response = 'y'
            logger.info("Auto-confirm: Starting embedding generation")
        else:
            response = input("Continue with embedding generation? [y/N]: ").strip().lower()

        if response != 'y':
            logger.info("Skipping embedding generation")
        else:
            try:
                with get_session() as session:
                    stats = generate_credit_event_embeddings(
                        session,
                        batch_size=100,
                        use_anthropic=None,  # Auto-detect
                        limit=args.limit  # Process all or limited
                    )

                    print()
                    logger.info(f"[OK] Embedding generation complete!")
                    logger.info(f"     Processed: {stats['processed']:,}")
                    logger.info(f"     Skipped: {stats['skipped']:,}")
                    logger.info(f"     Model: {stats['model']}")
            except KeyboardInterrupt:
                logger.info("\n[INTERRUPTED] Embedding generation stopped by user")
                logger.info("You can resume by running this script again")
            except Exception as e:
                logger.error(f"[ERROR] Failed to generate embeddings: {e}")
                import traceback
                traceback.print_exc()
                return 1
        print()

    # Step 4: Verify setup
    print("Step 3: Verifying setup...")
    print("-" * 70)
    try:
        if not verify_setup():
            return 1
    except Exception as e:
        logger.error(f"[ERROR] Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    print()

    # Step 5: Test retrieval
    print("Step 4: Testing retrieval functions...")
    print("-" * 70)
    try:
        test_retrieval()
    except Exception as e:
        logger.error(f"[ERROR] Retrieval test failed: {e}")
        import traceback
        traceback.print_exc()

    print()
    print("=" * 70)
    print("  [SUCCESS] Vector search setup complete!")
    print("=" * 70)
    print()
    print("You can now use the retrieval functions:")
    print("  from src.rag.retriever import vector_retrieve, sql_retrieve, hybrid_retrieve")
    print()
    print("Example usage:")
    print("  from src.db.session import get_session")
    print("  from src.rag.retriever import hybrid_retrieve")
    print()
    print("  with get_session() as session:")
    print('      results = hybrid_retrieve("bankruptcy filings", session, top_k=10)')
    print("      for r in results:")
    print('          print(r["company_name"], r["action_name"])')
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
