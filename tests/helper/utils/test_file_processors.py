"""
Tests for file processing utilities in helper.utils.file_processors.

This module tests:
- process_json_file() JSON file processing
- process_text_file() text file processing
- Document chunking and metadata creation
- ChromaDB collection integration

Why: File processors handle data extraction and chunking for vector database ingestion.
Args: File paths, ChromaDB collections, JSON/text content.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json
from pathlib import Path
from helper.utils.file_processors import process_json_file, process_text_file


class TestProcessJsonFile:
    """
    Test process_json_file() function.
    
    Tests: JSON parsing, document creation, metadata assignment, collection addition.
    Why: JSON files contain structured astrological data that needs chunking.
    Args: JSON file path, ChromaDB collection.
    """
    
    @patch('builtins.open', create=True)
    @patch('helper.utils.file_processors.create_metadata')
    def test_process_json_file_success(self, mock_create_metadata, mock_open):
        """
        Test process_json_file() successfully processes JSON file.
        
        What: Validates that JSON data is parsed and documents are created.
        Why: Ensures JSON files are correctly ingested into ChromaDB.
        Args: Valid JSON file path, mock collection.
        """
        json_data = {
            "Aries": {
                "traits": ["bold", "adventurous"],
                "element": "Fire"
            }
        }
        
        mock_file = MagicMock()
        mock_file.__enter__.return_value.read.return_value = json.dumps(json_data)
        mock_open.return_value = mock_file
        
        mock_collection = Mock()
        mock_create_metadata.return_value = {"zodiacs": "Aries"}
        
        process_json_file("test.json", mock_collection)
        
        assert mock_collection.add.called
        call_kwargs = mock_collection.add.call_args[1]
        assert 'documents' in call_kwargs
        assert len(call_kwargs['documents']) == 2  # Two key-value pairs
    
    @patch('builtins.open', create=True)
    def test_process_json_file_file_not_found(self, mock_open):
        """
        Test process_json_file() raises FileNotFoundError for missing file.
        
        What: Validates error handling when file doesn't exist.
        Why: Ensures clear error for missing files.
        Args: Non-existent file path.
        """
        mock_open.side_effect = FileNotFoundError("File not found")
        mock_collection = Mock()
        
        with pytest.raises(FileNotFoundError):
            process_json_file("nonexistent.json", mock_collection)
    
    @patch('builtins.open', create=True)
    def test_process_json_file_invalid_json(self, mock_open):
        """
        Test process_json_file() raises JSONDecodeError for invalid JSON.
        
        What: Validates error handling for malformed JSON.
        Why: Ensures invalid JSON is caught early.
        Args: File with invalid JSON content.
        """
        mock_file = MagicMock()
        mock_file.__enter__.return_value.read.return_value = "invalid json {"
        mock_open.return_value = mock_file
        
        mock_collection = Mock()
        
        with pytest.raises(json.JSONDecodeError):
            process_json_file("invalid.json", mock_collection)
    
    @patch('builtins.open', create=True)
    @patch('helper.utils.file_processors.create_metadata')
    def test_process_json_file_empty_data(self, mock_create_metadata, mock_open):
        """
        Test process_json_file() handles empty JSON data.
        
        What: Validates that empty JSON doesn't cause errors.
        Why: Ensures function handles edge cases gracefully.
        Args: JSON file with empty dict.
        """
        json_data = {}
        
        mock_file = MagicMock()
        mock_file.__enter__.return_value.read.return_value = json.dumps(json_data)
        mock_open.return_value = mock_file
        
        mock_collection = Mock()
        mock_create_metadata.return_value = {}
        
        process_json_file("empty.json", mock_collection)
        
        # Should not add any documents
        assert not mock_collection.add.called or len(mock_collection.add.call_args[1]['documents']) == 0


class TestProcessTextFile:
    """
    Test process_text_file() function.
    
    Tests: Text file reading, sentence splitting, document creation, collection addition.
    Why: Text files contain guidance content that needs sentence-level chunking.
    Args: Text file path, ChromaDB collection.
    """
    
    @patch('builtins.open', create=True)
    @patch('helper.utils.file_processors.create_metadata')
    def test_process_text_file_success(self, mock_create_metadata, mock_open):
        """
        Test process_text_file() successfully processes text file.
        
        What: Validates that text is split into sentences and documents are created.
        Why: Ensures text files are correctly ingested into ChromaDB.
        Args: Valid text file path, mock collection.
        """
        text_content = "First sentence.\nSecond sentence.\nThird sentence."
        
        mock_file = MagicMock()
        mock_file.__enter__.return_value.read.return_value = text_content
        mock_open.return_value = mock_file
        
        mock_collection = Mock()
        mock_create_metadata.return_value = {"life_areas": "love"}
        
        process_text_file("test.txt", mock_collection)
        
        assert mock_collection.add.called
        call_kwargs = mock_collection.add.call_args[1]
        assert 'documents' in call_kwargs
        assert len(call_kwargs['documents']) == 3  # Three sentences
    
    @patch('builtins.open', create=True)
    def test_process_text_file_file_not_found(self, mock_open):
        """
        Test process_text_file() raises FileNotFoundError for missing file.
        
        What: Validates error handling when file doesn't exist.
        Why: Ensures clear error for missing files.
        Args: Non-existent file path.
        """
        mock_open.side_effect = FileNotFoundError("File not found")
        mock_collection = Mock()
        
        with pytest.raises(FileNotFoundError):
            process_text_file("nonexistent.txt", mock_collection)
    
    @patch('builtins.open', create=True)
    @patch('helper.utils.file_processors.create_metadata')
    def test_process_text_file_empty_file(self, mock_create_metadata, mock_open):
        """
        Test process_text_file() handles empty text file.
        
        What: Validates that empty files don't cause errors.
        Why: Ensures function handles edge cases gracefully.
        Args: Empty text file.
        """
        mock_file = MagicMock()
        mock_file.__enter__.return_value.read.return_value = ""
        mock_open.return_value = mock_file
        
        mock_collection = Mock()
        mock_create_metadata.return_value = {}
        
        process_text_file("empty.txt", mock_collection)
        
        # Should not add any documents
        assert not mock_collection.add.called or len(mock_collection.add.call_args[1]['documents']) == 0
    
    @patch('builtins.open', create=True)
    @patch('helper.utils.file_processors.create_metadata')
    def test_process_text_file_filters_empty_lines(self, mock_create_metadata, mock_open):
        """
        Test process_text_file() filters out empty lines.
        
        What: Validates that empty lines are not processed as documents.
        Why: Avoids creating empty documents in ChromaDB.
        Args: Text file with empty lines.
        """
        text_content = "First sentence.\n\n\nSecond sentence.\n"
        
        mock_file = MagicMock()
        mock_file.__enter__.return_value.read.return_value = text_content
        mock_open.return_value = mock_file
        
        mock_collection = Mock()
        mock_create_metadata.return_value = {}
        
        process_text_file("test.txt", mock_collection)
        
        call_kwargs = mock_collection.add.call_args[1]
        # Should only have 2 documents (empty lines filtered)
        assert len(call_kwargs['documents']) == 2

