"""
Tests for data ingestion functions in helper.data_ingestion.

This module tests:
- ingest_data() main ingestion function
- Data directory validation
- JSON and text file processing
- ChromaDB collection initialization

Why: Data ingestion is critical for populating the vector database with astrological content.
Args: Data directory paths, collection names, recreate flags, file paths.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from helper.data_ingestion import ingest_data


class TestIngestData:
    """
    Test ingest_data() function.
    
    Tests: Complete ingestion flow, file processing, error handling.
    Why: Main function that orchestrates data loading into ChromaDB.
    Args: Data directory path, collection name, recreate flag.
    """
    
    @patch('helper.data_ingestion.init_chroma_db')
    @patch('helper.data_ingestion.get_openai_embedding_function')
    @patch('helper.data_ingestion.process_json_file')
    @patch('helper.data_ingestion.process_text_file')
    @patch('pathlib.Path.exists', return_value=True)
    @patch('pathlib.Path.absolute')
    def test_ingest_data_success(
        self,
        mock_absolute,
        mock_exists,
        mock_process_text,
        mock_process_json,
        mock_get_embedding,
        mock_init_chroma
    ):
        """
        Test ingest_data() successfully processes all files.
        
        What: Validates that JSON and text files are processed and added to collection.
        Why: Ensures data ingestion completes successfully.
        Args: Valid data directory with JSON and text files.
        """
        mock_collection = Mock()
        mock_collection.get.return_value = {'ids': ['doc1', 'doc2']}
        mock_init_chroma.return_value = mock_collection
        mock_get_embedding.return_value = Mock()
        
        # Mock Path.glob to return test files
        mock_absolute.return_value = Path("./data")
        with patch('pathlib.Path.glob') as mock_glob:
            # Create mock Path objects that can be converted to strings
            json_path1 = Mock()
            json_path1.stem = "test1"
            json_path1.__str__ = Mock(return_value="test1.json")
            json_path2 = Mock()
            json_path2.stem = "test2"
            json_path2.__str__ = Mock(return_value="test2.json")
            txt_path1 = Mock()
            txt_path1.stem = "test1"
            txt_path1.__str__ = Mock(return_value="test1.txt")
            txt_path2 = Mock()
            txt_path2.stem = "test2"
            txt_path2.__str__ = Mock(return_value="test2.txt")
            
            mock_glob.side_effect = [
                [json_path1, json_path2],  # JSON files
                [txt_path1, txt_path2]     # Text files
            ]
            
            ingest_data(data_directory="./data", collection_name="test_collection", recreate=True)
            
            assert mock_get_embedding.called
            assert mock_init_chroma.called
            assert mock_process_json.call_count == 2
            assert mock_process_text.call_count == 2
    
    @patch('helper.data_ingestion.init_chroma_db')
    @patch('helper.data_ingestion.get_openai_embedding_function')
    def test_ingest_data_directory_not_found(self, mock_get_embedding, mock_init_chroma):
        """
        Test ingest_data() raises ValueError when directory doesn't exist.
        
        What: Validates error handling for missing data directory.
        Why: Ensures clear error message for invalid paths.
        Args: Non-existent data directory path.
        """
        mock_get_embedding.return_value = Mock()
        mock_init_chroma.return_value = Mock()
        
        with patch('pathlib.Path.exists', return_value=False):
            with pytest.raises(ValueError) as exc_info:
                ingest_data(data_directory="./nonexistent", collection_name="test")
            
            assert "does not exist" in str(exc_info.value)
    
    @patch('helper.data_ingestion.init_chroma_db')
    @patch('helper.data_ingestion.get_openai_embedding_function')
    @patch('helper.data_ingestion.process_json_file')
    @patch('helper.data_ingestion.process_text_file')
    @patch('pathlib.Path.exists', return_value=True)
    @patch('pathlib.Path.absolute')
    def test_ingest_data_no_json_files(
        self,
        mock_absolute,
        mock_exists,
        mock_process_text,
        mock_process_json,
        mock_get_embedding,
        mock_init_chroma
    ):
        """
        Test ingest_data() handles directory with no JSON files.
        
        What: Validates that function continues when no JSON files found.
        Why: Ensures function handles empty directories gracefully.
        Args: Data directory with only text files.
        """
        mock_collection = Mock()
        mock_collection.get.return_value = {'ids': []}
        mock_init_chroma.return_value = mock_collection
        mock_get_embedding.return_value = Mock()
        
        mock_absolute.return_value = Path("./data")
        with patch('pathlib.Path.glob') as mock_glob:
            txt_path = Mock()
            txt_path.stem = "test"
            txt_path.__str__ = Mock(return_value="test.txt")
            mock_glob.side_effect = [
                [],  # No JSON files
                [txt_path]  # One text file
            ]
            
            ingest_data(data_directory="./data", collection_name="test")
            
            assert mock_process_json.call_count == 0
            assert mock_process_text.call_count == 1
    
    @patch('helper.data_ingestion.init_chroma_db')
    @patch('helper.data_ingestion.get_openai_embedding_function')
    @patch('helper.data_ingestion.process_json_file')
    @patch('helper.data_ingestion.process_text_file')
    @patch('pathlib.Path.exists', return_value=True)
    @patch('pathlib.Path.absolute')
    def test_ingest_data_no_text_files(
        self,
        mock_absolute,
        mock_exists,
        mock_process_text,
        mock_process_json,
        mock_get_embedding,
        mock_init_chroma
    ):
        """
        Test ingest_data() handles directory with no text files.
        
        What: Validates that function continues when no text files found.
        Why: Ensures function handles directories with only JSON files.
        Args: Data directory with only JSON files.
        """
        mock_collection = Mock()
        mock_collection.get.return_value = {'ids': []}
        mock_init_chroma.return_value = mock_collection
        mock_get_embedding.return_value = Mock()
        
        mock_absolute.return_value = Path("./data")
        with patch('pathlib.Path.glob') as mock_glob:
            json_path = Mock()
            json_path.stem = "test"
            json_path.__str__ = Mock(return_value="test.json")
            mock_glob.side_effect = [
                [json_path],  # One JSON file
                []  # No text files
            ]
            
            ingest_data(data_directory="./data", collection_name="test")
            
            assert mock_process_json.call_count == 1
            assert mock_process_text.call_count == 0
    
    @patch('helper.data_ingestion.init_chroma_db')
    @patch('helper.data_ingestion.get_openai_embedding_function')
    def test_ingest_data_recreate_collection(self, mock_get_embedding, mock_init_chroma):
        """
        Test ingest_data() recreates collection when recreate=True.
        
        What: Validates that collection is recreated when requested.
        Why: Allows fresh data ingestion by clearing existing collection.
        Args: recreate=True flag.
        """
        mock_collection = Mock()
        mock_collection.get.return_value = {'ids': []}
        mock_init_chroma.return_value = mock_collection
        mock_get_embedding.return_value = Mock()
        
        with patch('pathlib.Path.glob', return_value=[]):
            ingest_data(data_directory="./data", collection_name="test", recreate=True)
            
            mock_init_chroma.assert_called_once_with(
                "test",
                recreate=True,
                embedding_function=mock_get_embedding.return_value
            )

