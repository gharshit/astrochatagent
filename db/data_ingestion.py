"""
Data ingestion service for loading and processing astrological data into ChromaDB.

This module orchestrates the ingestion process by:
1. Loading OpenAI embeddings from environment variables
2. Initializing ChromaDB collection with custom embeddings
3. Processing JSON and text files from the data directory
4. Storing documents with appropriate metadata
"""

from pathlib import Path
from init_chroma_db import init_chroma_db
from utils import (
    get_openai_embedding_function,
    process_json_file,
    process_text_file,
    logger
)


def ingest_data(
    data_directory: str = "./data",
    collection_name: str = "astro_docs",
    recreate: bool = True
) -> None:
    """
    Main function to ingest all data from the data directory into ChromaDB.

    Process flow:
    1. Get OpenAI embedding function from environment variables
    2. Initialize Chroma collection with OpenAI embeddings
    3. Process all JSON files (chunked by key-value pairs)
    4. Process all text files (split by sentences)
    5. Store documents with appropriate metadata

    Args:
        data_directory (str): Path to data directory relative to script location
        collection_name (str): Name of Chroma collection
        recreate (bool): If True, deletes and recreates the collection
        
    Raises:
        ValueError: If data directory does not exist
    """
    logger.info("=" * 60)
    logger.info("Starting data ingestion process")
    logger.info("=" * 60)

    # Step 1: Get OpenAI embedding function
    logger.info("\n\n Step 1: Initializing OpenAI embedding function...")
    embedding_function = get_openai_embedding_function()

    # Step 2: Initialize Chroma collection with OpenAI embeddings
    logger.info(f"\n\n Step 2: Initializing Chroma collection '{collection_name}'...")
    collection = init_chroma_db(
        collection_name,
        recreate=recreate,
        embedding_function=embedding_function
    )

    # Step 3: Validate data directory
    logger.info(f"\n\n Step 3: Validating data directory: {data_directory}")
    data_path = Path(data_directory)
    if not data_path.exists():
        logger.error(f"Data directory does not exist: {data_path.absolute()}")
        raise ValueError(f"Data directory {data_directory} does not exist")
    
    logger.info(f"----> Data directory found: {data_path.absolute()}")

    # Step 4: Process all JSON files
    logger.info("\n\n Step 4: Processing JSON files...")
    json_files = list(data_path.glob("*.json"))
    logger.info(f"Found {len(json_files)} JSON file(s)")
    
    if not json_files:
        logger.warning("No JSON files found in data directory")
    else:
        for json_file in json_files:
            process_json_file(str(json_file), collection)

    # Step 5: Process all text files
    logger.info("\n\n Step 5: Processing text files...")
    text_files = list(data_path.glob("*.txt"))
    logger.info(f"-----> Found {len(text_files)} text file(s)")
    
    if not text_files:
        logger.warning("No text files found in data directory")
    else:
        for text_file in text_files:
            process_text_file(str(text_file), collection)

    # Summary
    total_docs = len(collection.get()['ids'])
    logger.info("=" * 60)
    logger.info("Data ingestion completed successfully")
    logger.info(f"Collection: '{collection_name}'")
    logger.info(f"Total documents: {total_docs}")
    logger.info("=" * 60)


if __name__ == "__main__":
    # Run data ingestion
    ingest_data()
