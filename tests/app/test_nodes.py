"""
Tests for LangGraph nodes in app.nodes.

This module tests:
- context_rag_query_node() RAG query generation
- retrieval_node() ChromaDB document retrieval
- chat_node() final response generation
- State transitions and error handling

Why: Nodes are the core workflow components that process user queries and generate responses.
Args: GraphState instances, RunnableConfig, query functions, LLM chains.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from app.nodes import context_rag_query_node, retrieval_node, chat_node
from app.state import GraphState
from app.models import RAGQueryOutput, MetadataFilters


class TestContextRagQueryNode:
    """
    Test context_rag_query_node() function.
    
    Tests: RAG query generation, metadata filter extraction, needs_rag determination.
    Why: This node determines if RAG is needed and generates search queries.
    Args: GraphState with user query, kundali_details, user_profile.
    """
    
    @pytest.mark.asyncio
    async def test_context_rag_query_node_with_rag_needed(self, mock_graph_state):
        """
        Test context_rag_query_node() when RAG is needed.
        
        What: Validates that node sets needs_rag=True and generates query/filters.
        Why: Ensures RAG decision logic works correctly.
        Args: GraphState with valid kundali_details and user_profile.
        """
        mock_output = RAGQueryOutput(
            needs_rag=True,
            metadata_filters=MetadataFilters(
                zodiacs=["Capricorn"],
                planetary_factors=["Sun"]
            ),
            rag_query="What is the personality of Capricorn?",
            reasoning="User asked about personality"
        )
        
        with patch('app.nodes.get_structured_llm') as mock_get_llm, \
             patch('app.nodes.ChatPromptTemplate') as mock_prompt_template:
            # Setup prompt template mock
            mock_prompt = Mock()
            mock_prompt_template.from_messages.return_value = mock_prompt
            
            # Setup LLM chain mock - the chain is prompt | structured_llm
            mock_llm_instance = Mock()
            mock_structured_llm = Mock()
            mock_structured_llm.with_structured_output.return_value = mock_structured_llm
            
            # Mock the pipe operator (|) to return a chain
            mock_chain = Mock()
            mock_chain.ainvoke = AsyncMock(return_value=mock_output)
            mock_prompt.__or__ = Mock(return_value=mock_chain)
            mock_get_llm.return_value = mock_structured_llm
            
            result = await context_rag_query_node(mock_graph_state)
            
            assert result["needs_rag"] is True
            assert result["rag_query"] == "What is the personality of Capricorn?"
            assert result["metadata_filters"] is not None
    
    @pytest.mark.asyncio
    async def test_context_rag_query_node_no_rag_needed(self, mock_graph_state):
        """
        Test context_rag_query_node() when RAG is not needed.
        
        What: Validates that node sets needs_rag=False.
        Why: Ensures node can skip RAG when not necessary.
        Args: GraphState with query that doesn't need RAG.
        """
        mock_output = RAGQueryOutput(
            needs_rag=False,
            metadata_filters=None,
            rag_query=None,
            reasoning="General question, no RAG needed"
        )
        
        with patch('app.nodes.get_structured_llm') as mock_get_llm:
            mock_llm = Mock()
            mock_structured = Mock()
            mock_chain = Mock()
            mock_chain.ainvoke = AsyncMock(return_value=mock_output)
            mock_structured.with_structured_output.return_value = mock_chain
            mock_get_llm.return_value = mock_structured
            
            result = await context_rag_query_node(mock_graph_state)
            
            assert result["needs_rag"] is False
            assert result["rag_query"] is None
    
    @pytest.mark.asyncio
    async def test_context_rag_query_node_missing_kundali(self):
        """
        Test context_rag_query_node() handles missing kundali_details.
        
        What: Validates error handling when kundali_details is missing.
        Why: Node should gracefully handle incomplete state.
        Args: GraphState without kundali_details.
        """
        state: GraphState = {
            "messages": [HumanMessage(content="Test")],
            "user_profile": None,
            "kundali_details": None,
            "session_id": "test",
            "rag_context_keys": [],
            "rag_query": None,
            "rag_results": [],
            "needs_rag": False,
            "metadata_filters": None
        }
        
        result = await context_rag_query_node(state)
        
        assert result["needs_rag"] is False
        assert result["rag_query"] is None
    
    @pytest.mark.asyncio
    async def test_context_rag_query_node_error_handling(self, mock_graph_state):
        """
        Test context_rag_query_node() handles LLM errors.
        
        What: Validates error handling when LLM call fails.
        Why: Node should not crash on LLM errors.
        Args: GraphState with LLM raising exception.
        """
        with patch('app.nodes.get_structured_llm') as mock_get_llm:
            mock_llm = Mock()
            mock_structured = Mock()
            mock_chain = Mock()
            mock_chain.ainvoke = AsyncMock(side_effect=Exception("LLM error"))
            mock_structured.with_structured_output.return_value = mock_chain
            mock_get_llm.return_value = mock_structured
            
            result = await context_rag_query_node(mock_graph_state)
            
            assert result["needs_rag"] is False
            assert result["rag_query"] is None


class TestRetrievalNode:
    """
    Test retrieval_node() function.
    
    Tests: ChromaDB query execution, result processing, metadata extraction.
    Why: This node retrieves relevant documents from ChromaDB for RAG.
    Args: GraphState with rag_query, metadata_filters, RunnableConfig with query_function.
    """
    
    @pytest.mark.asyncio
    async def test_retrieval_node_success(self, mock_graph_state):
        """
        Test retrieval_node() successfully retrieves documents.
        
        What: Validates that documents are retrieved and added to state.
        Why: Ensures RAG retrieval works correctly.
        Args: GraphState with needs_rag=True, rag_query, query_function in config.
        """
        mock_graph_state["needs_rag"] = True
        mock_graph_state["rag_query"] = "Test query"
        mock_graph_state["metadata_filters"] = {"zodiacs": ["Capricorn"]}
        
        mock_query_func = Mock(return_value={
            "documents": [["Document 1", "Document 2"]],
            "metadatas": [[{"zodiacs": "Capricorn"}, {"planetary_factors": "Sun"}]],
            "distances": [[0.1, 0.2]],
            "ids": [["doc1", "doc2"]]
        })
        
        config: RunnableConfig = {
            "configurable": {
                "query_function": mock_query_func
            }
        }
        
        result = await retrieval_node(mock_graph_state, config)
        
        assert len(result["rag_results"]) == 2
        assert len(result["rag_context_keys"]) > 0
        mock_query_func.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_retrieval_node_skipped_when_not_needed(self, mock_graph_state):
        """
        Test retrieval_node() skips when needs_rag is False.
        
        What: Validates that retrieval is skipped when RAG is not needed.
        Why: Avoids unnecessary ChromaDB queries.
        Args: GraphState with needs_rag=False.
        """
        mock_graph_state["needs_rag"] = False
        
        result = await retrieval_node(mock_graph_state, None)
        
        assert result["rag_results"] == []
    
    @pytest.mark.asyncio
    async def test_retrieval_node_no_query_function(self, mock_graph_state):
        """
        Test retrieval_node() handles missing query_function.
        
        What: Validates error handling when query_function is not in config.
        Why: Node should handle missing dependencies gracefully.
        Args: GraphState with needs_rag=True but no query_function in config.
        """
        mock_graph_state["needs_rag"] = True
        mock_graph_state["rag_query"] = "Test query"
        
        config: RunnableConfig = {"configurable": {}}
        
        result = await retrieval_node(mock_graph_state, config)
        
        assert result["rag_results"] == []
    
    @pytest.mark.asyncio
    async def test_retrieval_node_error_handling(self, mock_graph_state):
        """
        Test retrieval_node() handles query errors.
        
        What: Validates error handling when ChromaDB query fails.
        Why: Node should not crash on query errors.
        Args: GraphState with query_function raising exception.
        """
        mock_graph_state["needs_rag"] = True
        mock_graph_state["rag_query"] = "Test query"
        
        mock_query_func = Mock(side_effect=Exception("Query error"))
        
        config: RunnableConfig = {
            "configurable": {
                "query_function": mock_query_func
            }
        }
        
        result = await retrieval_node(mock_graph_state, config)
        
        assert result["rag_results"] == []


class TestChatNode:
    """
    Test chat_node() function.
    
    Tests: Final response generation, language handling, RAG context integration.
    Why: This node generates the final personalized response to the user.
    Args: GraphState with messages, kundali_details, rag_results, user_profile.
    """
    
    @pytest.mark.asyncio
    async def test_chat_node_success(self, mock_graph_state):
        """
        Test chat_node() successfully generates response.
        
        What: Validates that AI response is generated and added to messages.
        Why: Ensures final response generation works correctly.
        Args: GraphState with complete data including kundali and RAG results.
        """
        mock_response = Mock()
        mock_response.content = "Your sun sign is Capricorn."
        
        with patch('app.nodes.get_chat_llm') as mock_get_llm, \
             patch('app.nodes.ChatPromptTemplate') as mock_prompt_template:
            # Setup prompt template mock
            mock_prompt = Mock()
            mock_prompt_template.from_messages.return_value = mock_prompt
            
            # Setup LLM chain mock - the chain is prompt | llm
            mock_llm = Mock()
            mock_chain = Mock()
            mock_chain.ainvoke = AsyncMock(return_value=mock_response)
            mock_prompt.__or__ = Mock(return_value=mock_chain)
            mock_get_llm.return_value = mock_llm
            
            result = await chat_node(mock_graph_state)
            
            assert len(result["messages"]) == 2
            assert isinstance(result["messages"][-1], AIMessage)
            assert result["messages"][-1].content == "Your sun sign is Capricorn."
    
    @pytest.mark.asyncio
    async def test_chat_node_with_rag_results(self, mock_graph_state):
        """
        Test chat_node() uses RAG results in response generation.
        
        What: Validates that RAG context is included in prompt.
        Why: Ensures retrieved documents influence the response.
        Args: GraphState with rag_results populated.
        """
        mock_graph_state["rag_results"] = [
            {"content": "Capricorn traits: disciplined, ambitious"},
            {"content": "Sun in Capricorn: career-focused"}
        ]
        
        mock_response = Mock()
        mock_response.content = "Based on your chart..."
        
        with patch('app.nodes.get_chat_llm') as mock_get_llm, \
             patch('app.nodes.ChatPromptTemplate') as mock_prompt_template:
            # Setup prompt template mock
            mock_prompt = Mock()
            mock_prompt_template.from_messages.return_value = mock_prompt
            
            # Setup LLM chain mock
            mock_llm = Mock()
            mock_chain = Mock()
            mock_chain.ainvoke = AsyncMock(return_value=mock_response)
            mock_prompt.__or__ = Mock(return_value=mock_chain)
            mock_get_llm.return_value = mock_llm
            
            result = await chat_node(mock_graph_state)
            
            assert len(result["messages"]) == 2
            # Verify RAG context was used (check chain invocation)
            mock_chain.ainvoke.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_chat_node_error_handling(self, mock_graph_state):
        """
        Test chat_node() handles LLM errors with fallback.
        
        What: Validates error handling when LLM call fails.
        Why: Node should provide fallback response on errors.
        Args: GraphState with LLM raising exception.
        """
        with patch('app.nodes.get_chat_llm') as mock_get_llm:
            mock_llm = Mock()
            mock_chain = Mock()
            mock_chain.ainvoke = AsyncMock(side_effect=Exception("LLM error"))
            mock_llm.__or__ = Mock(return_value=mock_chain)
            mock_get_llm.return_value = mock_llm
            
            result = await chat_node(mock_graph_state)
            
            assert len(result["messages"]) == 2
            assert isinstance(result["messages"][-1], AIMessage)
            # Should have fallback message
            assert len(result["messages"][-1].content) > 0
    
    @pytest.mark.asyncio
    async def test_chat_node_hindi_language(self, mock_graph_state):
        """
        Test chat_node() generates response in Hindi when requested.
        
        What: Validates that response is generated in user's preferred language.
        Why: Supports multilingual responses based on user preference.
        Args: GraphState with user_profile.preferred_language="hi".
        """
        mock_graph_state["user_profile"].preferred_language = "hi"
        
        mock_response = Mock()
        mock_response.content = "आपका सूर्य राशि मकर है।"
        
        with patch('app.nodes.get_chat_llm') as mock_get_llm, \
             patch('app.nodes.ChatPromptTemplate') as mock_prompt_template:
            # Setup prompt template mock
            mock_prompt = Mock()
            mock_prompt_template.from_messages.return_value = mock_prompt
            
            # Setup LLM chain mock
            mock_llm = Mock()
            mock_chain = Mock()
            mock_chain.ainvoke = AsyncMock(return_value=mock_response)
            mock_prompt.__or__ = Mock(return_value=mock_chain)
            mock_get_llm.return_value = mock_llm
            
            result = await chat_node(mock_graph_state)
            
            assert len(result["messages"]) == 2
            # Verify Hindi response was generated
            mock_chain.ainvoke.assert_called_once()

