"""
Main FastAPI application with lifespan management.

Initializes embedding function, ChromaDB collection, query function,
and LangGraph checkpoint memory during application startup.
"""

##? Imports
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI
from dotenv import load_dotenv

load_dotenv()

from geopy.geocoders import Nominatim
from helper.utils import get_openai_embedding_function
from helper.data_ingestion import ingest_data
from helper.init_chroma_db import create_query_function, init_chroma_db
from langgraph.checkpoint.memory import MemorySaver
from vedicastro.VedicAstro import VedicHoroscopeData
from app.router import chat_router, kundali_router, session_router
from app.builder import compile_graph
from helper.utils import logger




@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan context manager for FastAPI application.
    
    Initializes all required components on startup:
    1. OpenAI embedding function
    2. ChromaDB collection
    3. Query function
    4. LangGraph checkpoint memory
    5. Compiled graph
    
    Cleans up on shutdown.
    """
    logger.info("=" * 60)
    logger.info("Starting application lifespan...")
    logger.info("=" * 60)
    
    try:
        # Step 0: Initialize Nominatim geocoder
        logger.info("Initializing Nominatim geocoder...")
        geocoder = Nominatim(user_agent="mynakshpoc")
        app.state.geocoder = geocoder
        logger.info("✓ Nominatim geocoder initialized")
        
        # Step 1: Initialize OpenAI embedding function
        logger.info("Initializing OpenAI embedding function...")
        embedding_function = get_openai_embedding_function()
        app.state.embedding_function = embedding_function
        logger.info("✓ Embedding function initialized")
        
        # Step 2: Initialize ChromaDB collection
        logger.info("Initializing ChromaDB collection...")
        collection = init_chroma_db(
            collection_name="astro_docs",
            recreate=False,
            embedding_function=embedding_function
        )
        app.state.chroma_collection = collection
        logger.info("✓ ChromaDB collection initialized")
        
        # Step 3: Create query function
        logger.info("Creating ChromaDB query function...")
        query_function = create_query_function(collection)
        app.state.query_function = query_function
        logger.info("✓ Query function created")
        
        # Step 4: Initialize LangGraph checkpoint memory
        logger.info("Initializing LangGraph checkpoint memory...")
        checkpoint_memory = MemorySaver()
        app.state.checkpoint_memory = checkpoint_memory
        logger.info("✓ Checkpoint memory initialized")
        
        # Step 5: Compile graph with checkpoint memory
        # When compiled with checkpoint_memory, LangGraph automatically:
        # - Persists state for each thread_id (session_id)
        # - Restores state when the same thread_id is used
        # - Manages conversation history across requests
        logger.info("Compiling LangGraph with checkpoint memory...")
        compiled_graph = compile_graph(checkpoint_memory)
        app.state.compiled_graph = compiled_graph
        logger.info("✓ Graph compiled successfully with checkpoint memory")
        
        logger.info("=" * 60)
        logger.info("Application initialization completed successfully")
        logger.info("=" * 60)
        
        yield
        
    except Exception as e:
        logger.error(f"Error during application initialization: {e}")
        raise
    finally:
        logger.info("Shutting down application...")
        # Cleanup if needed
        logger.info("Application shutdown complete")


# Create FastAPI app with lifespan
app = FastAPI(
    title="MyNakshpoc API",
    description="Vedic astrology knowledge base with LangGraph",
    version="0.1.0",
    lifespan=lifespan
)

# Include routers
app.include_router(chat_router)
app.include_router(kundali_router)
app.include_router(session_router)

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "MyNakshpoc API",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        dict: Health status with component availability
    """
    return {
        "status": "healthy",
        "components": {
            "embedding_function": hasattr(app.state, 'embedding_function') and app.state.embedding_function is not None,
            "chroma_collection": hasattr(app.state, 'chroma_collection') and app.state.chroma_collection is not None,
            "query_function": hasattr(app.state, 'query_function') and app.state.query_function is not None,
            "checkpoint_memory": hasattr(app.state, 'checkpoint_memory') and app.state.checkpoint_memory is not None,
            "compiled_graph": hasattr(app.state, 'compiled_graph') and app.state.compiled_graph is not None,
            "geocoder": hasattr(app.state, 'geocoder') and app.state.geocoder is not None,
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
