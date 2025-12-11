"""
Tests for chat router endpoints in app.router.chat_router.

This module tests:
- POST /v1/chat/ endpoint
- Session state management with LangGraph checkpoints
- Kundali details fetching and caching
- Error handling and HTTP status codes

Why: Chat router is the main API endpoint for user conversations.
Args: ChatRequest with session_id, message, user_profile, FastAPI Request.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from fastapi import HTTPException, status
from langchain_core.messages import HumanMessage, AIMessage
from app.router.chat_router import chat
from app.models import ChatRequest, ChatResponse
from app.state import GraphState


class TestChatEndpoint:
    """
    Test POST /v1/chat/ endpoint.
    
    Tests: Successful chat flow, state persistence, error handling.
    Why: Main chat endpoint that handles user conversations with state management.
    Args: ChatRequest, FastAPI Request with compiled_graph and checkpoint_memory.
    """
    
    @pytest.mark.asyncio
    async def test_chat_success_new_session(self, mock_user_profile, mock_fastapi_request):
        """
        Test chat endpoint successfully handles new session.
        
        What: Validates that new session creates state and generates response.
        Why: Ensures new conversations start correctly.
        Args: ChatRequest with new session_id, mock compiled_graph.
        """
        chat_request = ChatRequest(
            session_id="new_session_123",
            message="What is my sun sign?",
            user_profile=mock_user_profile
        )
        
        mock_graph_state: GraphState = {
            "messages": [AIMessage(content="Your sun sign is Capricorn.")],
            "user_profile": mock_user_profile,
            "kundali_details": Mock(),
            "session_id": "new_session_123",
            "rag_context_keys": ["zodiacs:Capricorn"],
            "rag_query": None,
            "rag_results": [],
            "needs_rag": False,
            "metadata_filters": None
        }
        
        with patch('app.router.chat_router.fetch_kundali_details', new_callable=AsyncMock) as mock_fetch:
            from app.models import KundaliDetails, KeyPositions, PlanetaryPosition
            # Create proper mock structure for kundali_details with all required attributes
            mock_sun = Mock(spec=PlanetaryPosition)
            mock_sun.sign = "Capricorn"
            mock_moon = Mock(spec=PlanetaryPosition)
            mock_moon.sign = "Leo"
            mock_ascendant = Mock(spec=PlanetaryPosition)
            mock_ascendant.sign = "Aries"
            mock_key_positions = Mock(spec=KeyPositions)
            mock_key_positions.sun = mock_sun
            mock_key_positions.moon = mock_moon
            mock_key_positions.ascendant = mock_ascendant
            mock_kundali = Mock(spec=KundaliDetails)
            mock_kundali.key_positions = mock_key_positions
            mock_fetch.return_value = mock_kundali
            
            mock_graph_state["kundali_details"] = mock_kundali
            mock_fastapi_request.app.state.compiled_graph.ainvoke = AsyncMock(return_value=mock_graph_state)
            mock_fastapi_request.app.state.checkpoint_memory.aget = AsyncMock(return_value=None)
            
            result = await chat(chat_request, mock_fastapi_request)
            
            assert isinstance(result, ChatResponse)
            assert result.response == "Your sun sign is Capricorn."
            assert len(result.context_used) > 0
            mock_fetch.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_chat_success_existing_session(self, mock_user_profile, mock_fastapi_request, mock_kundali_details):
        """
        Test chat endpoint uses existing session state.
        
        What: Validates that existing session reuses kundali_details from checkpoint.
        Why: Avoids redundant kundali calculations for same session.
        Args: ChatRequest with existing session_id, checkpoint with kundali_details.
        """
        chat_request = ChatRequest(
            session_id="existing_session_123",
            message="Tell me more about my moon sign",
            user_profile=mock_user_profile
        )
        
        mock_graph_state: GraphState = {
            "messages": [AIMessage(content="Your moon sign is Leo.")],
            "user_profile": mock_user_profile,
            "kundali_details": mock_kundali_details,
            "session_id": "existing_session_123",
            "rag_context_keys": [],
            "rag_query": None,
            "rag_results": [],
            "needs_rag": False,
            "metadata_filters": None
        }
        
        checkpoint_data = {
            "channel_values": {
                "kundali_details": mock_kundali_details.model_dump()
            }
        }
        
        mock_fastapi_request.app.state.compiled_graph.ainvoke = AsyncMock(return_value=mock_graph_state)
        mock_fastapi_request.app.state.checkpoint_memory.aget = AsyncMock(return_value=checkpoint_data)
        
        with patch('app.router.chat_router.fetch_kundali_details', new_callable=AsyncMock) as mock_fetch:
            result = await chat(chat_request, mock_fastapi_request)
            
            assert isinstance(result, ChatResponse)
            # Should not fetch kundali again
            mock_fetch.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_chat_missing_compiled_graph(self, mock_user_profile, mock_fastapi_request):
        """
        Test chat endpoint raises 503 when compiled_graph is missing.
        
        What: Validates error handling when LangGraph is not initialized.
        Why: Ensures proper service unavailable response.
        Args: FastAPI Request without compiled_graph in app.state.
        """
        chat_request = ChatRequest(
            session_id="test_session",
            message="Test message",
            user_profile=mock_user_profile
        )
        
        delattr(mock_fastapi_request.app.state, 'compiled_graph')
        
        with pytest.raises(HTTPException) as exc_info:
            await chat(chat_request, mock_fastapi_request)
        
        assert exc_info.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert "LangGraph service not available" in exc_info.value.detail
    
    @pytest.mark.asyncio
    async def test_chat_kundali_fetch_error(self, mock_user_profile, mock_fastapi_request):
        """
        Test chat endpoint handles kundali fetch errors.
        
        What: Validates error handling when kundali calculation fails.
        Why: Ensures errors propagate correctly.
        Args: ChatRequest with invalid user_profile causing kundali error.
        """
        chat_request = ChatRequest(
            session_id="test_session",
            message="Test message",
            user_profile=mock_user_profile
        )
        
        mock_fastapi_request.app.state.compiled_graph = Mock()
        mock_fastapi_request.app.state.checkpoint_memory.aget = AsyncMock(return_value=None)
        
        with patch('app.router.chat_router.fetch_kundali_details', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.side_effect = HTTPException(status_code=400, detail="Invalid birth date")
            
            with pytest.raises(HTTPException) as exc_info:
                await chat(chat_request, mock_fastapi_request)
            
            assert exc_info.value.status_code == 400
    
    @pytest.mark.asyncio
    async def test_chat_graph_execution_error(self, mock_user_profile, mock_fastapi_request):
        """
        Test chat endpoint handles graph execution errors.
        
        What: Validates error handling when LangGraph execution fails.
        Why: Ensures errors are caught and returned properly.
        Args: ChatRequest with compiled_graph raising exception.
        """
        chat_request = ChatRequest(
            session_id="test_session",
            message="Test message",
            user_profile=mock_user_profile
        )
        
        mock_fastapi_request.app.state.compiled_graph.ainvoke = AsyncMock(side_effect=Exception("Graph error"))
        mock_fastapi_request.app.state.checkpoint_memory.aget = AsyncMock(return_value=None)
        
        with patch('app.router.chat_router.fetch_kundali_details', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = Mock()
            
            with pytest.raises(HTTPException) as exc_info:
                await chat(chat_request, mock_fastapi_request)
            
            assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

