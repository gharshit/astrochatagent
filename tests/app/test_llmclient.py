"""
Tests for LLM client functions in app.llmclient.

This module tests:
- get_chat_llm() function with default and custom parameters
- get_structured_llm() function with default and custom parameters
- Environment variable handling and defaults
- Temperature and model name configuration

Why: Ensures LLM clients are correctly configured for chat and structured outputs.
Args: Model names, temperature values, environment variable configurations.
"""

import pytest
from unittest.mock import patch, MagicMock
from langchain_openai import ChatOpenAI
from app.llmclient import get_chat_llm, get_structured_llm, DEFAULT_CHAT_MODEL, DEFAULT_STRUCTURED_MODEL


class TestGetChatLLM:
    """
    Test get_chat_llm() function.
    
    Tests: Default model/temperature, custom parameters, environment variable handling.
    Why: Chat LLM is used for generating conversational responses.
    Args: Model name strings, temperature floats, environment variables.
    """
    
    @patch('app.llmclient.ChatOpenAI')
    @patch('app.llmclient.DEFAULT_CHAT_TEMPERATURE', 0.7)
    def test_get_chat_llm_defaults(self, mock_chat_openai):
        """
        Test get_chat_llm() with default parameters.
        
        What: Validates that default model and temperature are used when not specified.
        Why: Ensures function works with default configuration.
        Args: No arguments (uses defaults).
        """
        mock_instance = MagicMock()
        mock_chat_openai.return_value = mock_instance
        
        result = get_chat_llm()
        
        mock_chat_openai.assert_called_once()
        call_kwargs = mock_chat_openai.call_args[1]
        assert call_kwargs['model'] == DEFAULT_CHAT_MODEL
        assert call_kwargs['temperature'] == 0.7
        assert isinstance(result, MagicMock)
    
    @patch('app.llmclient.ChatOpenAI')
    @patch('app.llmclient.DEFAULT_CHAT_TEMPERATURE', 0.7)
    def test_get_chat_llm_custom_model(self, mock_chat_openai):
        """
        Test get_chat_llm() with custom model name.
        
        What: Validates that custom model name is used when provided.
        Why: Allows flexibility in model selection.
        Args: Custom model name string.
        """
        mock_instance = MagicMock()
        mock_chat_openai.return_value = mock_instance
        
        result = get_chat_llm(model="gpt-4")
        
        call_kwargs = mock_chat_openai.call_args[1]
        assert call_kwargs['model'] == "gpt-4"
        assert call_kwargs['temperature'] == 0.7
    
    @patch('app.llmclient.ChatOpenAI')
    def test_get_chat_llm_custom_temperature(self, mock_chat_openai):
        """
        Test get_chat_llm() with custom temperature.
        
        What: Validates that custom temperature is used when provided.
        Why: Allows control over response creativity.
        Args: Custom temperature float value.
        """
        mock_instance = MagicMock()
        mock_chat_openai.return_value = mock_instance
        
        result = get_chat_llm(temperature=0.9)
        
        call_kwargs = mock_chat_openai.call_args[1]
        assert call_kwargs['temperature'] == 0.9
    
    @patch('app.llmclient.ChatOpenAI')
    def test_get_chat_llm_custom_both(self, mock_chat_openai):
        """
        Test get_chat_llm() with both custom model and temperature.
        
        What: Validates that both custom parameters are used together.
        Why: Ensures function handles multiple custom parameters correctly.
        Args: Custom model name and temperature.
        """
        mock_instance = MagicMock()
        mock_chat_openai.return_value = mock_instance
        
        result = get_chat_llm(model="gpt-4", temperature=0.5)
        
        call_kwargs = mock_chat_openai.call_args[1]
        assert call_kwargs['model'] == "gpt-4"
        assert call_kwargs['temperature'] == 0.5


class TestGetStructuredLLM:
    """
    Test get_structured_llm() function.
    
    Tests: Default model/temperature, custom parameters, lower temperature for structured output.
    Why: Structured LLM is used for deterministic structured outputs (RAG decisions).
    Args: Model name strings, temperature floats.
    """
    
    @patch('app.llmclient.ChatOpenAI')
    def test_get_structured_llm_defaults(self, mock_chat_openai):
        """
        Test get_structured_llm() with default parameters.
        
        What: Validates that default structured model and temperature (0) are used.
        Why: Structured outputs need deterministic (low temperature) responses.
        Args: No arguments (uses defaults).
        """
        mock_instance = MagicMock()
        mock_chat_openai.return_value = mock_instance
        
        result = get_structured_llm()
        
        call_kwargs = mock_chat_openai.call_args[1]
        assert call_kwargs['model'] == DEFAULT_STRUCTURED_MODEL
        assert call_kwargs['temperature'] == 0.0
    
    @patch('app.llmclient.ChatOpenAI')
    def test_get_structured_llm_custom_model(self, mock_chat_openai):
        """
        Test get_structured_llm() with custom model name.
        
        What: Validates that custom model name is used when provided.
        Why: Allows flexibility in structured model selection.
        Args: Custom model name string.
        """
        mock_instance = MagicMock()
        mock_chat_openai.return_value = mock_instance
        
        result = get_structured_llm(model="gpt-4")
        
        call_kwargs = mock_chat_openai.call_args[1]
        assert call_kwargs['model'] == "gpt-4"
        assert call_kwargs['temperature'] == 0.0
    
    @patch('app.llmclient.ChatOpenAI')
    def test_get_structured_llm_custom_temperature(self, mock_chat_openai):
        """
        Test get_structured_llm() with custom temperature.
        
        What: Validates that custom temperature can override default (0).
        Why: Allows fine-tuning structured output determinism.
        Args: Custom temperature float value.
        """
        mock_instance = MagicMock()
        mock_chat_openai.return_value = mock_instance
        
        result = get_structured_llm(temperature=0.1)
        
        call_kwargs = mock_chat_openai.call_args[1]
        assert call_kwargs['temperature'] == 0.1

