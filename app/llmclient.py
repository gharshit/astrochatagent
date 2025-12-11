"""
LLM client instances for LangGraph nodes.
"""

from langchain_openai import ChatOpenAI
from helper.utils.logger import setup_logger
import os
from dotenv import load_dotenv

load_dotenv()

logger = setup_logger(name="app.llmclient", level=20)  # INFO level

# Load model configuration from environment variables
DEFAULT_CHAT_MODEL            = os.getenv("LLM_CHAT_MODEL", "gpt-4o-mini")
DEFAULT_STRUCTURED_MODEL      = os.getenv("LLM_STRUCTURED_MODEL", "gpt-4o-mini")
DEFAULT_CHAT_TEMPERATURE      = float(os.getenv("LLM_CHAT_TEMPERATURE", "0.7"))
DEFAULT_STRUCTURED_TEMPERATURE = float(os.getenv("LLM_STRUCTURED_TEMPERATURE", "0"))


def get_chat_llm(model: str | None = None, temperature: float | None = None) -> ChatOpenAI:
    """
    Get a standard ChatOpenAI instance for general chat operations.
    
    Args:
        model: Model name (defaults to LLM_CHAT_MODEL env var or "gpt-4o-mini")
        temperature: Temperature setting (defaults to LLM_CHAT_TEMPERATURE env var or 0.7)
        
    Returns:
        ChatOpenAI instance
    """
    model_name = model or DEFAULT_CHAT_MODEL
    temp_value = temperature if temperature is not None else DEFAULT_CHAT_TEMPERATURE
    
    logger.debug(f"Creating chat LLM: model={model_name}, temperature={temp_value}")
    return ChatOpenAI(model=model_name, temperature=temp_value)


def get_structured_llm(model: str | None = None, temperature: float | None = None) -> ChatOpenAI:
    """
    Get a ChatOpenAI instance optimized for structured output.
    
    Uses lower temperature for more deterministic structured outputs.
    
    Args:
        model: Model name (defaults to LLM_STRUCTURED_MODEL env var or "gpt-4o-mini")
        temperature: Temperature setting (defaults to LLM_STRUCTURED_TEMPERATURE env var or 0)
        
    Returns:
        ChatOpenAI instance configured for structured output
    """
    model_name = model or DEFAULT_STRUCTURED_MODEL
    temp_value = temperature if temperature is not None else DEFAULT_STRUCTURED_TEMPERATURE
    
    logger.debug(f"Creating structured LLM: model={model_name}, temperature={temp_value}")
    return ChatOpenAI(model=model_name, temperature=temp_value)

