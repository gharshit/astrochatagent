"""
LangGraph state definition for chat conversations.
"""

from typing import TypedDict, Annotated, List
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
    rag_context_keys: List[str]
    rag_query: str | None
    rag_results: List[str]
    

