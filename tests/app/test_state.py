"""
Tests for GraphState TypedDict in app.state.

This module tests:
- GraphState structure and required fields
- Type annotations and field types
- State initialization and field access

Why: GraphState defines the state structure for LangGraph workflow.
Args: Messages, user_profile, kundali_details, session_id, RAG-related fields.
"""

import pytest
from langchain_core.messages import HumanMessage, AIMessage
from app.state import GraphState
from app.models import UserProfile, KundaliDetails


class TestGraphState:
    """
    Test GraphState TypedDict structure.
    
    Tests: Field types, required fields, state initialization.
    Why: GraphState is the core data structure for LangGraph workflow.
    Args: All GraphState fields including messages, profiles, RAG data.
    """
    
    def test_graph_state_structure(self, mock_user_profile, mock_kundali_details):
        """
        Test creating a valid GraphState.
        
        What: Validates that GraphState can be created with all required fields.
        Why: Ensures state structure matches LangGraph requirements.
        Args: All GraphState fields with valid data.
        """
        state: GraphState = {
            "messages": [HumanMessage(content="Hello")],
            "user_profile": mock_user_profile,
            "kundali_details": mock_kundali_details,
            "session_id": "test_session",
            "rag_context_keys": [],
            "rag_query": None,
            "rag_results": [],
            "needs_rag": False,
            "metadata_filters": None
        }
        
        assert len(state["messages"]) == 1
        assert state["user_profile"] == mock_user_profile
        assert state["kundali_details"] == mock_kundali_details
        assert state["session_id"] == "test_session"
        assert state["needs_rag"] is False
    
    def test_graph_state_with_rag_data(self, mock_user_profile, mock_kundali_details):
        """
        Test GraphState with RAG-related fields populated.
        
        What: Validates that RAG fields can be set in state.
        Why: Ensures state can track RAG query and results.
        Args: RAG query string, results list, context keys, metadata filters.
        """
        state: GraphState = {
            "messages": [HumanMessage(content="What is my sun sign?")],
            "user_profile": mock_user_profile,
            "kundali_details": mock_kundali_details,
            "session_id": "test_session",
            "rag_context_keys": ["zodiacs:Capricorn", "planetary_factors:Sun"],
            "rag_query": "What is the sun sign for Capricorn?",
            "rag_results": [
                {"content": "Test document", "metadata": {"zodiacs": "Capricorn"}}
            ],
            "needs_rag": True,
            "metadata_filters": {"zodiacs": ["Capricorn"]}
        }
        
        assert state["needs_rag"] is True
        assert state["rag_query"] == "What is the sun sign for Capricorn?"
        assert len(state["rag_results"]) == 1
        assert len(state["rag_context_keys"]) == 2
        assert state["metadata_filters"] is not None
    
    def test_graph_state_with_multiple_messages(self, mock_user_profile, mock_kundali_details):
        """
        Test GraphState with multiple messages (conversation history).
        
        What: Validates that state can hold multiple messages.
        Why: LangGraph maintains conversation history in state.
        Args: Multiple HumanMessage and AIMessage objects.
        """
        state: GraphState = {
            "messages": [
                HumanMessage(content="Hello"),
                AIMessage(content="Hi! How can I help?"),
                HumanMessage(content="What is my sun sign?")
            ],
            "user_profile": mock_user_profile,
            "kundali_details": mock_kundali_details,
            "session_id": "test_session",
            "rag_context_keys": [],
            "rag_query": None,
            "rag_results": [],
            "needs_rag": False,
            "metadata_filters": None
        }
        
        assert len(state["messages"]) == 3
        assert isinstance(state["messages"][0], HumanMessage)
        assert isinstance(state["messages"][1], AIMessage)
        assert isinstance(state["messages"][2], HumanMessage)
    
    def test_graph_state_optional_fields_none(self, mock_user_profile):
        """
        Test GraphState with optional fields as None.
        
        What: Validates that optional fields can be None.
        Why: State may not always have kundali_details or RAG data.
        Args: State with kundali_details and RAG fields as None.
        """
        state: GraphState = {
            "messages": [HumanMessage(content="Hello")],
            "user_profile": mock_user_profile,
            "kundali_details": None,
            "session_id": "test_session",
            "rag_context_keys": [],
            "rag_query": None,
            "rag_results": [],
            "needs_rag": False,
            "metadata_filters": None
        }
        
        assert state["kundali_details"] is None
        assert state["rag_query"] is None
        assert state["metadata_filters"] is None

