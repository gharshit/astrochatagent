"""
LangGraph state definition for chat conversations.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph.message import add_messages
from app.models import UserProfile, KundaliDetails


class GraphState(TypedDict):
    """
    State for the LangGraph chat flow.
    
    This state is automatically managed by LangGraph's checkpoint system
    when using MemorySaver. Each thread_id (session_id) maintains its own state.
    """
    messages: Annotated[list, add_messages]
    user_profile: UserProfile | None
    kundali_details: KundaliDetails | None
    session_id: str
    rag_context_keys: List[str]  # Keys from metadata (zodiacs, planetary_factors, etc.)
    rag_query: str | None  # Generated query for embedding search
    rag_results: List[Dict[str, Any]]  # Retrieved documents with metadata
    needs_rag: bool  # Whether RAG is needed for this query
    metadata_filters: Dict[str, Any] | None  # Metadata filters for ChromaDB query
    

