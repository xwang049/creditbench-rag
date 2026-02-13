"""Demo script for Text-to-SQL RAG system.

This script demonstrates how to use the Text-to-SQL RAG system to ask
questions about the CreditBench database.

Requirements:
- Database running with data loaded
- ANTHROPIC_API_KEY in .env file
"""

from src.rag import sql_rag_answer, text_to_sql
from src.db.session import SessionLocal


def demo_basic_usage():
    """Basic usage example."""
    print("=" * 80)
    print("Example 1: Basic Question")
    print("=" * 80)

    question = "How many companies are in the database?"

    result = sql_rag_answer(question)

    if result["success"]:
        print(f"\n📊 Question: {result['question']}")
        print(f"\n🔍 Generated SQL:\n{result['sql']}")
        print(f"\n✅ Results: {result['row_count']} rows")
        print(f"\n💡 Answer:\n{result['answer']}\n")
    else:
        print(f"\n❌ Error: {result['error']}\n")


def demo_sql_generation_only():
    """Example of just generating SQL without executing."""
    print("=" * 80)
    print("Example 2: SQL Generation Only")
    print("=" * 80)

    questions = [
        "How many bankruptcy filings were there in 2023?",
        "Which technology companies had the lowest DTD in 2022?",
        "What was the 10-year Treasury yield on 2023-01-15?",
    ]

    for question in questions:
        try:
            sql = text_to_sql(question)
            print(f"\n📊 Question: {question}")
            print(f"🔍 SQL:\n{sql}\n")
        except Exception as e:
            print(f"\n❌ Error generating SQL: {e}\n")


def demo_multiple_questions():
    """Example of asking multiple questions."""
    print("=" * 80)
    print("Example 3: Multiple Questions")
    print("=" * 80)

    questions = [
        "How many credit events occurred in 2023?",
        "List the top 5 sectors by number of companies",
        "Show me recent bankruptcy filings (limit 5)",
    ]

    session = SessionLocal()

    try:
        for i, question in enumerate(questions, 1):
            print(f"\n{'='*60}")
            print(f"Question {i}: {question}")
            print('='*60)

            result = sql_rag_answer(question, session=session)

            if result["success"]:
                print(f"\n📊 SQL:\n{result['sql']}")
                print(f"\n💡 Answer:\n{result['answer']}\n")
            else:
                print(f"\n❌ Error: {result['error']}\n")

    finally:
        session.close()


def demo_with_error_handling():
    """Example with comprehensive error handling."""
    print("=" * 80)
    print("Example 4: Error Handling")
    print("=" * 80)

    # This question might fail if table doesn't exist or API key is missing
    question = "Show me data from nonexistent_table"

    result = sql_rag_answer(question)

    print(f"\n📊 Question: {question}")

    if result["success"]:
        print(f"✅ Success!")
        print(f"Answer: {result['answer']}")
    else:
        print(f"❌ Failed as expected")
        print(f"Error: {result['error']}")

        if result["sql"]:
            print(f"\nGenerated SQL (before error):\n{result['sql']}")


def demo_credit_risk_analysis():
    """Example of credit risk analysis questions."""
    print("=" * 80)
    print("Example 5: Credit Risk Analysis (DTD)")
    print("=" * 80)

    questions = [
        "What is the average DTD for companies in the Energy sector in 2023?",
        "Which companies have DTD below 2 (high default risk)?",
        "Show me DTD statistics by sector for the most recent month",
    ]

    print("\n📊 Distance-to-Default (DTD) is the key credit risk metric:")
    print("   - Higher DTD = Lower default risk (safer)")
    print("   - Lower DTD = Higher default risk (riskier)")
    print("   - DTD < 2 indicates high default probability\n")

    for question in questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print('='*60)

        result = sql_rag_answer(question)

        if result["success"]:
            print(f"\n🔍 SQL:\n{result['sql']}")
            print(f"\n💡 Answer:\n{result['answer']}\n")
        else:
            print(f"\n❌ Error: {result['error']}\n")


if __name__ == "__main__":
    import sys

    print("\n" + "="*80)
    print("CreditBench Text-to-SQL RAG Demo")
    print("="*80)
    print("\nThis demo shows how to use the Text-to-SQL RAG system.")
    print("Make sure you have:")
    print("  1. PostgreSQL running with CreditBench data loaded")
    print("  2. ANTHROPIC_API_KEY in your .env file")
    print("="*80 + "\n")

    # Check if we should run all examples or just one
    if len(sys.argv) > 1:
        example_num = int(sys.argv[1])
        examples = {
            1: demo_basic_usage,
            2: demo_sql_generation_only,
            3: demo_multiple_questions,
            4: demo_with_error_handling,
            5: demo_credit_risk_analysis,
        }

        if example_num in examples:
            examples[example_num]()
        else:
            print(f"Unknown example number: {example_num}")
            print("Available examples: 1-5")
    else:
        # Run all examples
        try:
            demo_basic_usage()
            input("\nPress Enter to continue to next example...")

            demo_sql_generation_only()
            input("\nPress Enter to continue to next example...")

            demo_multiple_questions()
            input("\nPress Enter to continue to next example...")

            demo_with_error_handling()
            input("\nPress Enter to continue to next example...")

            demo_credit_risk_analysis()

        except KeyboardInterrupt:
            print("\n\nDemo interrupted by user.")
        except Exception as e:
            print(f"\n\n❌ Error running demo: {e}")
            print("\nMake sure you have:")
            print("  1. Database running: psql $DATABASE_URL")
            print("  2. ANTHROPIC_API_KEY set in .env")
            print("  3. Data loaded: python -m src.ingestion.load_all")

    print("\n" + "="*80)
    print("Demo complete!")
    print("Try the interactive CLI: python -m src.rag.sql_retriever")
    print("="*80 + "\n")
