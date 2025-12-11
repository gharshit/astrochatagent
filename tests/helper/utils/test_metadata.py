"""
Tests for metadata creation utilities in helper.utils.metadata.

This module tests:
- create_metadata() function
- Metadata assignment based on file type
- Zodiac, planetary, nakshatra, and life area metadata

Why: Metadata is critical for filtering and organizing documents in ChromaDB.
Args: Filenames, main keys from JSON structure.
"""

import pytest
from helper.utils.metadata import create_metadata


class TestCreateMetadata:
    """
    Test create_metadata() function.
    
    Tests: Metadata creation for different file types, key extraction.
    Why: Ensures documents have correct metadata for filtering.
    Args: Filename, optional main_key from JSON structure.
    """
    
    def test_create_metadata_zodiac_traits(self):
        """
        Test create_metadata() creates zodiac metadata for zodiac_traits file.
        
        What: Validates that zodiac_traits files get zodiac metadata.
        Why: Enables filtering by zodiac sign in RAG queries.
        Args: Filename "zodiac_traits", main_key "Aries".
        """
        metadata = create_metadata("zodiac_traits", "Aries")
        
        assert metadata["zodiacs"] == "Aries"
        assert metadata["content_type"] == "zodiac_traits"
    
    def test_create_metadata_planetary_impact(self):
        """
        Test create_metadata() creates planetary metadata for planetary_impact file.
        
        What: Validates that planetary_impact files get planetary_factors metadata.
        Why: Enables filtering by planetary factors in RAG queries.
        Args: Filename "planetary_impact", main_key "Sun".
        """
        metadata = create_metadata("planetary_impact", "Sun")
        
        assert metadata["planetary_factors"] == "Sun"
        assert metadata["content_type"] == "planetary_impact"
    
    def test_create_metadata_nakshtras(self):
        """
        Test create_metadata() creates nakshatra metadata for nakshtras file.
        
        What: Validates that nakshtras files get nakshtra metadata.
        Why: Enables filtering by nakshatra in RAG queries.
        Args: Filename "nakshtras", main_key "Ashwini".
        """
        metadata = create_metadata("nakshtras", "Ashwini")
        
        assert metadata["nakshtra"] == "Ashwini"
        assert metadata["content_type"] == "nakshtras"
    
    def test_create_metadata_love_guidance(self):
        """
        Test create_metadata() creates life area metadata for love_guidance file.
        
        What: Validates that love_guidance files get life_areas="love" metadata.
        Why: Enables filtering by life area in RAG queries.
        Args: Filename "love_guidance", no main_key.
        """
        metadata = create_metadata("love_guidance")
        
        assert metadata["life_areas"] == "love"
        assert metadata["content_type"] == "life_guidance"
    
    def test_create_metadata_spiritual_guidance(self):
        """
        Test create_metadata() creates spirituality metadata for spiritual_guidance file.
        
        What: Validates that spiritual_guidance files get life_areas="spirituality" metadata.
        Why: Enables filtering by spirituality life area.
        Args: Filename "spiritual_guidance", no main_key.
        """
        metadata = create_metadata("spiritual_guidance")
        
        assert metadata["life_areas"] == "spirituality"
        assert metadata["content_type"] == "life_guidance"
    
    def test_create_metadata_career_guidance(self):
        """
        Test create_metadata() creates career metadata for carrer_guidance file.
        
        What: Validates that carrer_guidance files get life_areas="career" metadata.
        Why: Enables filtering by career life area.
        Args: Filename "carrer_guidance", no main_key.
        """
        metadata = create_metadata("carrer_guidance")
        
        assert metadata["life_areas"] == "career"
        assert metadata["content_type"] == "life_guidance"
    
    def test_create_metadata_unknown_file(self):
        """
        Test create_metadata() creates generic metadata for unknown files.
        
        What: Validates that unknown file types get generic metadata.
        Why: Ensures all files have at least content_type metadata.
        Args: Unknown filename, optional main_key.
        """
        metadata = create_metadata("unknown_file")
        
        assert metadata["content_type"] == "general"
        assert "zodiacs" not in metadata
        assert "planetary_factors" not in metadata
    
    def test_create_metadata_no_main_key(self):
        """
        Test create_metadata() handles missing main_key gracefully.
        
        What: Validates that function works when main_key is None.
        Why: Text files don't have main_key, should still get metadata.
        Args: Filename, main_key=None.
        """
        metadata = create_metadata("zodiac_traits", None)
        
        assert metadata["zodiacs"] == "general"
        assert metadata["content_type"] == "zodiac_traits"

