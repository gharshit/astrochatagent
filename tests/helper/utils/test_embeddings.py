"""
Tests for embedding utilities in helper.utils.embeddings.

This module tests:
- get_openai_embedding_function() function
- LangChainOpenAIEmbeddingFunction class
- Environment variable handling
- Embedding dimension validation

Why: Embeddings are critical for vector database operations and semantic search.
Args: API keys, model names, dimension values, environment variables.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import os
from helper.utils.embeddings import (
    get_openai_embedding_function,
    LangChainOpenAIEmbeddingFunction
)


class TestLangChainOpenAIEmbeddingFunction:
    """
    Test LangChainOpenAIEmbeddingFunction class.
    
    Tests: Initialization, embedding generation, dimension handling.
    Why: This class bridges LangChain embeddings with ChromaDB.
    Args: API key, model name, dimensions.
    """
    
    @patch('helper.utils.embeddings.OpenAIEmbeddings')
    def test_embedding_function_init_default(self, mock_openai_embeddings):
        """
        Test LangChainOpenAIEmbeddingFunction initialization with defaults.
        
        What: Validates that embedding function initializes with default model.
        Why: Ensures default configuration works correctly.
        Args: API key, default model and dimensions.
        """
        mock_embeddings_instance = Mock()
        mock_openai_embeddings.return_value = mock_embeddings_instance
        
        func = LangChainOpenAIEmbeddingFunction(
            api_key="test_key",
            model="text-embedding-3-small"
        )
        
        assert func.model == "text-embedding-3-small"
        assert func.dimensions is None
        mock_openai_embeddings.assert_called_once()
    
    @patch('helper.utils.embeddings.OpenAIEmbeddings')
    def test_embedding_function_init_with_dimensions(self, mock_openai_embeddings):
        """
        Test LangChainOpenAIEmbeddingFunction initialization with custom dimensions.
        
        What: Validates that embedding function accepts custom dimensions.
        Why: Allows control over embedding vector size.
        Args: API key, model name, custom dimensions value.
        """
        mock_embeddings_instance = Mock()
        mock_openai_embeddings.return_value = mock_embeddings_instance
        
        func = LangChainOpenAIEmbeddingFunction(
            api_key="test_key",
            model="text-embedding-3-small",
            dimensions=512
        )
        
        assert func.dimensions == 512
        call_kwargs = mock_openai_embeddings.call_args[1]
        assert call_kwargs.get('dimensions') == 512
    
    @patch('helper.utils.embeddings.OpenAIEmbeddings')
    def test_embedding_function_call(self, mock_openai_embeddings):
        """
        Test LangChainOpenAIEmbeddingFunction __call__ method.
        
        What: Validates that embedding function generates embeddings for input texts.
        Why: Ensures function can embed documents for ChromaDB.
        Args: List of text strings.
        """
        mock_embeddings_instance = Mock()
        mock_embeddings_instance.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        mock_openai_embeddings.return_value = mock_embeddings_instance
        
        func = LangChainOpenAIEmbeddingFunction(api_key="test_key")
        result = func(["test document"])
        
        assert len(result) == 1
        assert len(result[0]) == 3
        mock_embeddings_instance.embed_documents.assert_called_once_with(["test document"])


class TestGetOpenAIEmbeddingFunction:
    """
    Test get_openai_embedding_function() function.
    
    Tests: Environment variable reading, function creation, error handling.
    Why: Main function for creating embedding functions from environment config.
    Args: Environment variables (OPENAI_API_KEY, LLM_EMBEDDING_MODEL, etc.).
    """
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    @patch('helper.utils.embeddings.LangChainOpenAIEmbeddingFunction')
    def test_get_openai_embedding_function_success(self, mock_embedding_class):
        """
        Test get_openai_embedding_function() successfully creates function.
        
        What: Validates that function is created from environment variables.
        Why: Ensures embedding function can be initialized from config.
        Args: OPENAI_API_KEY in environment.
        """
        mock_instance = Mock()
        mock_instance.return_value = [[0.1, 0.2]]
        mock_embedding_class.return_value = mock_instance
        
        result = get_openai_embedding_function()
        
        assert result is not None
        mock_embedding_class.assert_called_once()
    
    @patch.dict(os.environ, {}, clear=True)
    def test_get_openai_embedding_function_missing_api_key(self):
        """
        Test get_openai_embedding_function() raises ValueError when API key missing.
        
        What: Validates error handling when OPENAI_API_KEY is not set.
        Why: Ensures clear error message for missing configuration.
        Args: No OPENAI_API_KEY in environment.
        """
        with pytest.raises(ValueError) as exc_info:
            get_openai_embedding_function()
        
        assert "OPENAI_API_KEY" in str(exc_info.value)
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key', 'LLM_EMBEDDING_MODEL': 'gpt-4'})
    @patch('helper.utils.embeddings.LangChainOpenAIEmbeddingFunction')
    def test_get_openai_embedding_function_custom_model(self, mock_embedding_class):
        """
        Test get_openai_embedding_function() uses custom model from environment.
        
        What: Validates that custom model name is read from environment.
        Why: Allows model selection via environment variables.
        Args: LLM_EMBEDDING_MODEL environment variable.
        """
        mock_instance = Mock()
        mock_embedding_class.return_value = mock_instance
        
        result = get_openai_embedding_function()
        
        call_kwargs = mock_embedding_class.call_args[1]
        assert call_kwargs['model'] == 'gpt-4'
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key', 'LLM_EMBEDDING_DIMENSIONS': '512'})
    @patch('helper.utils.embeddings.LangChainOpenAIEmbeddingFunction')
    def test_get_openai_embedding_function_custom_dimensions(self, mock_embedding_class):
        """
        Test get_openai_embedding_function() uses custom dimensions from environment.
        
        What: Validates that custom dimensions are read from environment.
        Why: Allows dimension control via environment variables.
        Args: LLM_EMBEDDING_DIMENSIONS environment variable.
        """
        mock_instance = Mock()
        mock_instance.return_value = [[0.1] * 512]
        mock_embedding_class.return_value = mock_instance
        
        result = get_openai_embedding_function()
        
        call_kwargs = mock_embedding_class.call_args[1]
        assert call_kwargs['dimensions'] == 512
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    @patch('helper.utils.embeddings.LangChainOpenAIEmbeddingFunction')
    def test_get_openai_embedding_function_dimension_verification(self, mock_embedding_class):
        """
        Test get_openai_embedding_function() verifies embedding dimensions.
        
        What: Validates that function tests embedding dimensions after creation.
        Why: Ensures dimensions match expected values.
        Args: Embedding function that returns test embeddings.
        """
        mock_instance = Mock()
        mock_instance.return_value = [[0.1] * 1536]  # Default dimension
        mock_embedding_class.return_value = mock_instance
        
        result = get_openai_embedding_function()
        
        # Function should test embedding
        assert mock_instance.called

