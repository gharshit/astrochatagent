"""
Embedding function utilities using LangChain and OpenAI.
"""

import os
from typing import List
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from chromadb.api.types import EmbeddingFunction, Embeddings
from .logger import logger

# Load environment variables
load_dotenv()


class LangChainOpenAIEmbeddingFunction(EmbeddingFunction):
    """
    ChromaDB-compatible embedding function using LangChain's OpenAI embeddings.
    
    This class implements ChromaDB's EmbeddingFunction interface by wrapping
    LangChain's OpenAIEmbeddings.
    """
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        """
        Initialize the embedding function.
        
        Args:
            api_key (str): OpenAI API key
            model (str): OpenAI embedding model name
        """
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=api_key,
            model=model
        )
    
    def __call__(self, input: List[str]) -> Embeddings:
        """
        Generate embeddings for the given texts.
        
        Args:
            input (List[str]): List of texts to embed
            
        Returns:
            Embeddings: List of embedding vectors
        """
        return self.embeddings.embed_documents(input)


def get_openai_embedding_function():
    """
    Create OpenAI embedding function using LangChain.

    Returns:
        LangChainOpenAIEmbeddingFunction: ChromaDB compatible embedding function
        
    Raises:
        ValueError: If OPENAI_API_KEY is not found in environment variables
    """
    logger.info("Initializing OpenAI embedding function...")
    api_key = os.getenv("OPENAI_API_KEY")
    model   = os.getenv("OPENAI_MODEL", "text-embedding-3-small")

    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment variables")
        raise ValueError(
            "OPENAI_API_KEY not found in environment variables. "
            "Please create a .env file with OPENAI_API_KEY=your_key"
        )

    logger.info(f"Using OpenAI model: {model}")
    embedding_func = LangChainOpenAIEmbeddingFunction(api_key=api_key, model=model)
    logger.info("OpenAI embedding function initialized successfully")
    return embedding_func

