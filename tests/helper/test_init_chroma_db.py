"""
Tests for ChromaDB initialization functions in helper.init_chroma_db.

This module tests:
- init_chroma_db() collection initialization
- create_query_function() query function creation
- Collection creation, loading, and deletion
- Embedding function handling

Why: ChromaDB initialization is critical for vector database operations.
Args: Collection names, recreate flags, persist directories, embedding functions.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from helper.init_chroma_db import init_chroma_db, create_query_function


class TestInitChromaDB:
    """
    Test init_chroma_db() function.
    
    Tests: Collection creation, loading, deletion, embedding function handling.
    Why: Ensures ChromaDB collections are properly initialized.
    Args: Collection name, recreate flag, persist directory, embedding function.
    """
    
    @patch('helper.init_chroma_db.chromadb.PersistentClient')
    def test_init_chroma_db_create_new(self, mock_client_class):
        """
        Test init_chroma_db() creates new collection when it doesn't exist.
        
        What: Validates that new collection is created when not found.
        Why: Ensures function can create fresh collections.
        Args: Collection name that doesn't exist, recreate=False.
        """
        mock_client = Mock()
        mock_client.list_collections.return_value = []
        mock_client.create_collection.return_value = Mock()
        mock_client_class.return_value = mock_client
        
        mock_embedding = Mock()
        result = init_chroma_db("new_collection", recreate=False, embedding_function=mock_embedding)
        
        mock_client.create_collection.assert_called_once()
        assert result is not None
    
    @patch('helper.init_chroma_db.chromadb.PersistentClient')
    def test_init_chroma_db_recreate_existing(self, mock_client_class):
        """
        Test init_chroma_db() recreates collection when recreate=True.
        
        What: Validates that existing collection is deleted and recreated.
        Why: Allows fresh start by clearing existing data.
        Args: Existing collection name, recreate=True.
        """
        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.name = "existing_collection"
        mock_client.list_collections.return_value = [mock_collection]
        mock_client.delete_collection = Mock()
        mock_client.create_collection.return_value = Mock()
        mock_client_class.return_value = mock_client
        
        mock_embedding = Mock()
        result = init_chroma_db("existing_collection", recreate=True, embedding_function=mock_embedding)
        
        mock_client.delete_collection.assert_called_once_with(name="existing_collection")
        mock_client.create_collection.assert_called_once()
    
    @patch('helper.init_chroma_db.chromadb.PersistentClient')
    def test_init_chroma_db_load_existing(self, mock_client_class):
        """
        Test init_chroma_db() loads existing collection when recreate=False.
        
        What: Validates that existing collection is loaded without recreation.
        Why: Preserves existing data when not recreating.
        Args: Existing collection name, recreate=False.
        """
        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.name = "existing_collection"
        mock_client.list_collections.return_value = [mock_collection]
        mock_client.get_collection.return_value = Mock()
        mock_client_class.return_value = mock_client
        
        mock_embedding = Mock()
        result = init_chroma_db("existing_collection", recreate=False, embedding_function=mock_embedding)
        
        mock_client.get_collection.assert_called_once()
        mock_client.create_collection.assert_not_called()
    
    @patch('helper.init_chroma_db.chromadb.PersistentClient')
    def test_init_chroma_db_no_embedding_function(self, mock_client_class):
        """
        Test init_chroma_db() works without embedding function.
        
        What: Validates that function works with default embedding.
        Why: Supports collections that use default embeddings.
        Args: Collection name, embedding_function=None.
        """
        mock_client = Mock()
        mock_client.list_collections.return_value = []
        mock_client.create_collection.return_value = Mock()
        mock_client_class.return_value = mock_client
        
        result = init_chroma_db("test_collection", embedding_function=None)
        
        mock_client.create_collection.assert_called_once()
        # Should not pass embedding_function when None
        call_kwargs = mock_client.create_collection.call_args[1]
        assert 'embedding_function' in call_kwargs or call_kwargs.get('embedding_function') is None


class TestCreateQueryFunction:
    """
    Test create_query_function() function.
    
    Tests: Query function creation, ChromaDB query execution.
    Why: Query function is used for RAG retrieval from ChromaDB.
    Args: ChromaDB collection instance.
    """
    
    def test_create_query_function_success(self, mock_chroma_collection):
        """
        Test create_query_function() creates valid query function.
        
        What: Validates that query function is created and callable.
        Why: Ensures query function can execute ChromaDB queries.
        Args: Mock ChromaDB collection.
        """
        query_func = create_query_function(mock_chroma_collection)
        
        assert callable(query_func)
        
        # Test query execution
        result = query_func("test query", n_results=5)
        
        assert result is not None
        mock_chroma_collection.query.assert_called_once()
    
    def test_create_query_function_with_where_clause(self, mock_chroma_collection):
        """
        Test create_query_function() handles where clause filters.
        
        What: Validates that query function accepts metadata filters.
        Why: RAG queries need metadata filtering for relevant documents.
        Args: Mock collection, query text, where clause dict.
        """
        query_func = create_query_function(mock_chroma_collection)
        
        where_clause = {"zodiacs": {"$in": ["Capricorn"]}}
        result = query_func("test query", n_results=3, where=where_clause)
        
        call_kwargs = mock_chroma_collection.query.call_args[1]
        assert call_kwargs.get('where') == where_clause
    
    def test_create_query_function_custom_n_results(self, mock_chroma_collection):
        """
        Test create_query_function() accepts custom n_results.
        
        What: Validates that query function accepts different result counts.
        Why: Allows flexibility in number of retrieved documents.
        Args: Mock collection, query text, custom n_results value.
        """
        query_func = create_query_function(mock_chroma_collection)
        
        result = query_func("test query", n_results=10)
        
        call_kwargs = mock_chroma_collection.query.call_args[1]
        assert call_kwargs.get('n_results') == 10

