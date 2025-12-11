import chromadb
from typing import Optional, Callable
from .utils.logger import logger


def init_chroma_db(
    collection_name   : str,
    recreate          : bool = False,
    persist_directory : str = "./vector_db",
    embedding_function: Optional[Callable] = None
):
    """
    Initialize a Chroma DB collection with optional custom embedding function.

    Args:
        collection_name (str): Name of the collection to load/create.
        recreate (bool): If True, deletes the collection and recreates it.
        persist_directory (str): Path for persistent Chroma DB storage.
        embedding_function: Optional embedding function. If None, uses default.

    Returns:
        chromadb.api.types.Collection: The initialized Chroma collection.
    """
    # Initialize Chroma client with persistent storage
    client = chromadb.PersistentClient(path=persist_directory)

    # Fetch existing collections
    existing_collections = [c.name for c in client.list_collections()]

    if recreate:
        # Delete if exists
        if collection_name in existing_collections:
            logger.info(f"Collection '{collection_name}' found. Deleting & recreating...")
            client.delete_collection(name=collection_name)
        else:
            logger.info(f"Collection '{collection_name}' does not exist. Creating fresh.")

        # Create new collection with embedding function
        logger.debug(f"Creating collection '{collection_name}' with embedding function")
        collection = client.create_collection(
            name               = collection_name,
            embedding_function = embedding_function,
            metadata           = {"hnsw:space": "cosine"}  # cosine similarity
        )
        logger.info(f"✓ Collection '{collection_name}' created successfully")
        return collection

    # If not recreating, load or create
    if collection_name in existing_collections:
        logger.info(f"Collection '{collection_name}' exists. Loading...")
        return client.get_collection(collection_name)
    else:
        logger.info(f"Collection '{collection_name}' does not exist. Creating new one...")
        logger.debug(f"Creating collection '{collection_name}' with embedding function")
        return client.create_collection(
            name               = collection_name,
            embedding_function = embedding_function,
            metadata           = {"hnsw:space": "cosine"}
        )

## ? Create query function
def create_query_function(collection):
    """
    Create a query function for ChromaDB collection.
    
    Args:
        collection: ChromaDB collection instance
        
    Returns:
        Query function that takes query text and returns results
    """
    def query_chroma(query_text: str, n_results: int = 5, **kwargs):
        """
        Query ChromaDB collection.
        
        Args:
            query_text: Text to search for
            n_results: Number of results to return
            **kwargs: Additional query parameters (where, etc.)
            
        Returns:
            Query results with documents, metadatas, and distances
        """
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            **kwargs
        )
        return results
    
    return query_chroma

