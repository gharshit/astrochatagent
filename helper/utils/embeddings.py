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
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small", dimensions: int | None = None):
        """
        Initialize the embedding function.
        
        Args:
            api_key (str): OpenAI API key
            model (str): OpenAI embedding model name
            dimensions (int | None): Embedding dimensions. If None, uses model default.
                                    For text-embedding-3-small: default is 1536, can be 512, 256, etc.
                                    For text-embedding-3-large: default is 3072, can be 1024, 256, etc.
        """
        self.model = model
        self.dimensions = dimensions
        
        # Build kwargs for OpenAIEmbeddings
        embeddings_kwargs = {
            "openai_api_key": api_key,
            "model": model
        }
        
        # Only add dimensions if explicitly provided (some models don't support it)
        if dimensions is not None:
            embeddings_kwargs["dimensions"] = dimensions
            logger.info(f"Using explicit embedding dimensions: {dimensions}")
        
        self.embeddings = OpenAIEmbeddings(**embeddings_kwargs)
    
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
    
    Reads configuration from environment variables:
    - OPENAI_API_KEY: Required OpenAI API key
    - LLM_EMBEDDING_MODEL: Embedding model name (default: text-embedding-3-small)
    - LLM_EMBEDDING_DIMENSIONS: Embedding dimensions (optional, uses model default if not set)
                                Note: Must match the dimensions used when creating the ChromaDB collection.
                                Your collection expects 1536 dimensions.

    Returns:
        LangChainOpenAIEmbeddingFunction: ChromaDB compatible embedding function
        
    Raises:
        ValueError: If OPENAI_API_KEY is not found in environment variables
    """
    logger.info("Initializing OpenAI embedding function...")
    api_key = os.getenv("OPENAI_API_KEY")
    model   = os.getenv("LLM_EMBEDDING_MODEL", "text-embedding-3-small")
    
    # Parse dimensions from environment if provided
    dimensions_str = os.getenv("LLM_EMBEDDING_DIMENSIONS")
    dimensions = int(dimensions_str) if dimensions_str else None

    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment variables")
        raise ValueError(
            "OPENAI_API_KEY not found in environment variables. "
            "Please create a .env file with OPENAI_API_KEY=your_key"
        )

    logger.info(f"Using OpenAI model: {model}")
    if dimensions:
        logger.info(f"Using embedding dimensions: {dimensions}")
    else:
        logger.info("Using model default dimensions")
    
    embedding_func = LangChainOpenAIEmbeddingFunction(
        api_key=api_key,
        model=model,
        dimensions=dimensions
    )
    
    # Test embedding to verify actual dimensions
    try:
        test_embedding = embedding_func(["test"])
        actual_dimension = len(test_embedding[0]) if test_embedding else None
        if actual_dimension:
            logger.info(f"✓ Verified embedding dimension: {actual_dimension}")
            if dimensions and actual_dimension != dimensions:
                logger.warning(
                    f"Warning: Requested {dimensions} dimensions but got {actual_dimension}. "
                    f"This may cause issues with your ChromaDB collection."
                )
    except Exception as e:
        logger.warning(f"Could not verify embedding dimensions: {e}")
    
    logger.info("OpenAI embedding function initialized successfully")
    return embedding_func

