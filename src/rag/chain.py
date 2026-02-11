"""RAG chain for question answering over creditbench data."""

from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage

from src.config import settings
from src.rag.embeddings import EmbeddingService
from src.rag.retriever import VectorRetriever


class RAGChain:
    """RAG chain for answering questions about credit data."""

    SYSTEM_PROMPT = """You are an expert credit analyst with deep knowledge of credit events,
corporate finance, and macroeconomic indicators.

You have access to a database of companies, credit events (defaults, bankruptcies, downgrades, etc.),
and macroeconomic indicators.

When answering questions:
1. Base your answers on the provided context from the database
2. Be precise and cite specific data points when available
3. If the context doesn't contain enough information, say so
4. Provide numerical data and dates when relevant
5. Explain credit implications and risk factors clearly

Context from database:
{context}
"""

    USER_PROMPT = """Question: {question}

Please provide a clear, data-driven answer based on the context above."""

    def __init__(
        self,
        session: Session,
        model: Optional[str] = None,
        temperature: float = 0.1
    ):
        """Initialize RAG chain.

        Args:
            session: SQLAlchemy session
            model: LLM model name
            temperature: LLM temperature (lower = more deterministic)
        """
        self.session = session
        self.embeddings = EmbeddingService()
        self.retriever = VectorRetriever(session, self.embeddings)

        self.llm = ChatAnthropic(
            model=model or settings.LLM_MODEL,
            temperature=temperature,
            anthropic_api_key=settings.ANTHROPIC_API_KEY,
        )

    def _format_context(self, retrieved_data: Dict[str, Any]) -> str:
        """Format retrieved data into context string.

        Args:
            retrieved_data: Dictionary of retrieved entities

        Returns:
            Formatted context string
        """
        context_parts = []

        # Format companies
        if 'companies' in retrieved_data and retrieved_data['companies']:
            context_parts.append("## Relevant Companies:")
            for company in retrieved_data['companies']:
                company_text = [
                    f"- {company.name} ({company.ticker})",
                    f"  Sector: {company.sector or 'N/A'}",
                    f"  Industry: {company.industry or 'N/A'}",
                ]
                if hasattr(company, 'description') and company.description:
                    company_text.append(f"  Description: {company.description}")
                context_parts.append("\n".join(company_text))

        # Format credit events
        if 'credit_events' in retrieved_data and retrieved_data['credit_events']:
            context_parts.append("\n## Relevant Credit Events:")
            for event in retrieved_data['credit_events']:
                event_text = [
                    f"- Date: {event.event_date}",
                    f"  Type: {event.event_type}",
                    f"  Company ID: {event.company_id}",
                ]
                if hasattr(event, 'rating') and event.rating:
                    event_text.append(f"  Rating: {event.rating}")
                if hasattr(event, 'description') and event.description:
                    event_text.append(f"  Description: {event.description}")
                context_parts.append("\n".join(event_text))

        return "\n\n".join(context_parts) if context_parts else "No relevant data found."

    def query(
        self,
        question: str,
        retrieve_companies: bool = True,
        retrieve_events: bool = True,
        max_results: int = 5
    ) -> Dict[str, Any]:
        """Answer a question using RAG.

        Args:
            question: User question
            retrieve_companies: Whether to retrieve company data
            retrieve_events: Whether to retrieve credit event data
            max_results: Maximum results per entity type

        Returns:
            Dictionary with answer, sources, and metadata
        """
        # Retrieve relevant data
        retrieved_data = self.retriever.hybrid_search(
            query=question,
            limit=max_results,
            include_companies=retrieve_companies,
            include_events=retrieve_events
        )

        # Format context
        context = self._format_context(retrieved_data)

        # Build prompt
        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT.format(context=context)),
            HumanMessage(content=self.USER_PROMPT.format(question=question))
        ]

        # Get LLM response
        response = self.llm.invoke(messages)

        return {
            'answer': response.content,
            'sources': {
                'companies': [
                    {'id': c.id, 'name': c.name, 'ticker': c.ticker}
                    for c in retrieved_data.get('companies', [])
                ],
                'credit_events': [
                    {'id': e.id, 'event_type': e.event_type, 'event_date': str(e.event_date)}
                    for e in retrieved_data.get('credit_events', [])
                ]
            },
            'context': context,
            'question': question
        }

    def query_company(self, company_id: int, question: str) -> Dict[str, Any]:
        """Answer a question about a specific company.

        Args:
            company_id: Company ID
            question: User question

        Returns:
            Dictionary with answer and company context
        """
        # Get comprehensive company context
        company_context = self.retriever.get_company_context(company_id)

        if not company_context:
            return {
                'answer': f"Company with ID {company_id} not found.",
                'sources': {},
                'context': '',
                'question': question
            }

        # Format context
        context = self._format_context({
            'companies': [company_context['company']],
            'credit_events': company_context['credit_events']
        })

        # Build prompt
        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT.format(context=context)),
            HumanMessage(content=self.USER_PROMPT.format(question=question))
        ]

        # Get LLM response
        response = self.llm.invoke(messages)

        return {
            'answer': response.content,
            'company': {
                'id': company_context['company'].id,
                'name': company_context['company'].name,
                'ticker': company_context['company'].ticker
            },
            'sources': {
                'credit_events': [
                    {'id': e.id, 'event_type': e.event_type, 'event_date': str(e.event_date)}
                    for e in company_context['credit_events']
                ]
            },
            'context': context,
            'question': question
        }
