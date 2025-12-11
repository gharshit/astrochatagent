"""
Shared pytest fixtures and configuration for all tests.

This module provides common fixtures, mocks, and test utilities
that are used across multiple test files.
"""

import pytest
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage
from app.models import (
    UserProfile,
    KundaliDetails,
    BirthDetails,
    LocationDetails,
    ChartSettings,
    KeyPositions,
    PlanetaryPosition,
    PlanetData,
    HouseData,
    MetadataFilters,
    RAGQueryOutput
)
from app.state import GraphState


@pytest.fixture
def mock_user_profile() -> UserProfile:
    """
    Create a mock UserProfile for testing.
    
    Returns:
        UserProfile: Mock user profile with test data
    """
    return UserProfile(
        name="Test User",
        birth_date="1990-01-15",
        birth_time="10:30",
        birth_place="New Delhi, India",
        preffered_language="en"
    )


@pytest.fixture
def mock_kundali_details() -> KundaliDetails:
    """
    Create a mock KundaliDetails for testing.
    
    Returns:
        KundaliDetails: Mock kundali details with test astrological data
    """
    return KundaliDetails(
        user_name="Test User",
        birth_details=BirthDetails(
            birth_date="1990-01-15",
            birth_time="10:30",
            birth_place="New Delhi, India",
            year=1990,
            month=1,
            day=15,
            hour=10,
            minute=30,
            second=0
        ),
        location=LocationDetails(
            latitude=28.6139,
            longitude=77.2090,
            utc_offset="+05:30"
        ),
        chart_settings=ChartSettings(
            ayanamsa="Lahiri",
            house_system="Equal"
        ),
        key_positions=KeyPositions(
            sun=PlanetaryPosition(
                sign="Capricorn",
                nakshatra="Uttara Ashadha",
                nakshatra_pada=1,
                nakshatra_lord="Sun",
                rasi_lord="Saturn",
                sub_lord="Sun",
                sub_sub_lord="Sun",
                longitude=285.5
            ),
            moon=PlanetaryPosition(
                sign="Leo",
                nakshatra="Magha",
                nakshatra_pada=2,
                nakshatra_lord="Ketu",
                rasi_lord="Sun",
                sub_lord="Moon",
                sub_sub_lord="Moon",
                longitude=135.2
            ),
            ascendant=PlanetaryPosition(
                sign="Aries",
                nakshatra="Ashwini",
                nakshatra_pada=3,
                nakshatra_lord="Ketu",
                rasi_lord="Mars",
                sub_lord="Mars",
                sub_sub_lord="Mars",
                longitude=5.8
            ),
            lagna_lord="Mars"
        ),
        planets=[
            PlanetData(
                object="Sun",
                rasi="Capricorn",
                is_retrograde=False,
                longitude_dec_deg=285.5,
                sign_lon_dms="15°30'00\"",
                sign_lon_dec_deg=15.5,
                lat_dms="0°00'00\"",
                nakshatra="Uttara Ashadha",
                rasi_lord="Saturn",
                nakshatra_lord="Sun",
                sub_lord="Sun",
                sub_sub_lord="Sun",
                house_nr=10
            )
        ],
        houses=[
            HouseData(
                object="I",
                house_nr=1,
                rasi="Aries",
                longitude_dec_deg=5.8,
                sign_lon_dms="5°48'00\"",
                sign_lon_dec_deg=5.8,
                deg_size=30.0,
                nakshatra="Ashwini",
                rasi_lord="Mars",
                nakshatra_lord="Ketu",
                sub_lord="Mars",
                sub_sub_lord="Mars"
            )
        ],
        planetary_aspects=[],
        consolidated_chart=None,
        vimshottari_dasa={}
    )


@pytest.fixture
def mock_graph_state(mock_user_profile: UserProfile, mock_kundali_details: KundaliDetails) -> GraphState:
    """
    Create a mock GraphState for testing.
    
    Args:
        mock_user_profile: Mock user profile fixture
        mock_kundali_details: Mock kundali details fixture
        
    Returns:
        GraphState: Mock graph state with initial values
    """
    return {
        "messages": [HumanMessage(content="Test message")],
        "user_profile": mock_user_profile,
        "kundali_details": mock_kundali_details,
        "session_id": "test_session_123",
        "rag_context_keys": [],
        "rag_query": None,
        "rag_results": [],
        "needs_rag": False,
        "metadata_filters": None
    }


@pytest.fixture
def mock_rag_query_output() -> RAGQueryOutput:
    """
    Create a mock RAGQueryOutput for testing.
    
    Returns:
        RAGQueryOutput: Mock RAG query output with test filters
    """
    return RAGQueryOutput(
        needs_rag=True,
        metadata_filters=MetadataFilters(
            zodiacs=["Capricorn", "Leo"],
            planetary_factors=["Sun", "Moon"],
            life_areas=["career"],
            nakshtra=["Uttara Ashadha"]
        ),
        rag_query="What is the career guidance for Capricorn sun sign?",
        reasoning="User is asking about career guidance specific to their sun sign"
    )


@pytest.fixture
def mock_chroma_collection():
    """
    Create a mock ChromaDB collection for testing.
    
    Returns:
        Mock: Mock ChromaDB collection object
    """
    collection = Mock()
    collection.name = "test_collection"
    collection.get.return_value = {
        "ids": ["doc1", "doc2"],
        "documents": [["Test document 1"], ["Test document 2"]],
        "metadatas": [[{"zodiacs": "Capricorn"}], [{"planetary_factors": "Sun"}]]
    }
    collection.query.return_value = {
        "documents": [["Test document 1", "Test document 2"]],
        "metadatas": [[{"zodiacs": "Capricorn"}, {"planetary_factors": "Sun"}]],
        "distances": [[0.1, 0.2]],
        "ids": [["doc1", "doc2"]]
    }
    return collection


@pytest.fixture
def mock_query_function(mock_chroma_collection):
    """
    Create a mock query function for ChromaDB.
    
    Args:
        mock_chroma_collection: Mock ChromaDB collection fixture
        
    Returns:
        Callable: Mock query function
    """
    def query_func(query_text: str, n_results: int = 5, **kwargs):
        return mock_chroma_collection.query.return_value
    return query_func


@pytest.fixture
def mock_llm_chain():
    """
    Create a mock LLM chain for testing.
    
    Returns:
        AsyncMock: Mock async LLM chain
    """
    chain = AsyncMock()
    chain.ainvoke = AsyncMock(return_value=MagicMock(
        content="Test AI response",
        needs_rag=True,
        rag_query="Test query"
    ))
    return chain


@pytest.fixture
def mock_geocoder():
    """
    Create a mock geocoder for testing.
    
    Returns:
        Mock: Mock geocoder object
    """
    geocoder = Mock()
    location = Mock()
    location.latitude = 28.6139
    location.longitude = 77.2090
    geocoder.geocode.return_value = location
    return geocoder


@pytest.fixture
def mock_vedic_data():
    """
    Create a mock VedicHoroscopeData for testing.
    
    Returns:
        Mock: Mock VedicHoroscopeData object
    """
    vedic_data = Mock()
    chart = {
        "Sun": Mock(sign="Capricorn", lon=285.5),
        "Moon": Mock(sign="Leo", lon=135.2),
        "Asc": Mock(sign="Aries", lon=5.8)
    }
    vedic_data.generate_chart.return_value = chart
    vedic_data.get_planets_data_from_chart.return_value = []
    vedic_data.get_houses_data_from_chart.return_value = []
    vedic_data.get_planetary_aspects.return_value = []
    vedic_data.get_consolidated_chart_data.return_value = None
    vedic_data.compute_vimshottari_dasa.return_value = {}
    vedic_data.get_rl_nl_sl_data.return_value = {
        "Nakshatra": "Uttara Ashadha",
        "Pada": 1,
        "NakshatraLord": "Sun",
        "RasiLord": "Saturn",
        "SubLord": "Sun",
        "SubSubLord": "Sun"
    }
    vedic_data.ayanamsa = "Lahiri"
    vedic_data.house_system = "Equal"
    return vedic_data


@pytest.fixture
def mock_fastapi_request(mock_geocoder, mock_query_function):
    """
    Create a mock FastAPI Request object for testing.
    
    Args:
        mock_geocoder: Mock geocoder fixture
        mock_query_function: Mock query function fixture
        
    Returns:
        Mock: Mock FastAPI Request object
    """
    request = Mock()
    request.app.state.geocoder = mock_geocoder
    request.app.state.query_function = mock_query_function
    request.app.state.compiled_graph = Mock()
    request.app.state.checkpoint_memory = Mock()
    return request

