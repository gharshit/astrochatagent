"""
Tests for Pydantic models in app.models.

This module tests:
- UserProfile model validation (birth_date, birth_time formats)
- ChatRequest and ChatResponse models
- KundaliDetails and related models structure
- RAGQueryOutput and MetadataFilters models
- Field validators and type constraints

Why: Ensures data models correctly validate input and maintain type safety.
Args: Various test data including valid/invalid dates, times, and model structures.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError
from app.models import (
    UserProfile,
    ChatRequest,
    ChatResponse,
    KundaliDetails,
    BirthDetails,
    LocationDetails,
    ChartSettings,
    KeyPositions,
    PlanetaryPosition,
    PlanetData,
    HouseData,
    MetadataFilters,
    RAGQueryOutput,
    ZodiacSign,
    PlanetaryFactor,
    LifeArea,
    NakshatraName
)


class TestUserProfile:
    """
    Test UserProfile model validation.
    
    Tests: Birth date/time format validation, language constraints, field requirements.
    Why: UserProfile is a critical input model that must validate user data correctly.
    Args: Various date/time strings, language codes, and field combinations.
    """
    
    def test_valid_user_profile(self):
        """
        Test creating a valid UserProfile.
        
        What: Validates that a properly formatted UserProfile can be created.
        Why: Ensures the model accepts valid input data.
        Args: Valid name, birth_date (YYYY-MM-DD), birth_time (HH:MM), birth_place, language.
        """
        profile = UserProfile(
            name="John Doe",
            birth_date="1990-01-15",
            birth_time="10:30",
            birth_place="New Delhi, India",
            preffered_language="en"
        )
        assert profile.name == "John Doe"
        assert profile.birth_date == "1990-01-15"
        assert profile.birth_time == "10:30"
        assert profile.preffered_language == "en"
    
    def test_invalid_birth_date_format(self):
        """
        Test UserProfile with invalid birth date format.
        
        What: Validates that invalid date formats are rejected.
        Why: Prevents invalid date data from entering the system.
        Args: Invalid date strings (DD-MM-YYYY, MM/DD/YYYY, etc.).
        """
        with pytest.raises(ValidationError) as exc_info:
            UserProfile(
                name="John Doe",
                birth_date="15-01-1990",  # Wrong format
                birth_time="10:30",
                birth_place="New Delhi",
                preffered_language="en"
            )
        assert "Invalid birth date" in str(exc_info.value)
    
    def test_invalid_birth_time_format(self):
        """
        Test UserProfile with invalid birth time format.
        
        What: Validates that invalid time formats are rejected.
        Why: Ensures time data is in correct HH:MM format.
        Args: Invalid time strings (12-hour format, missing minutes, etc.).
        """
        with pytest.raises(ValidationError) as exc_info:
            UserProfile(
                name="John Doe",
                birth_date="1990-01-15",
                birth_time="10:30 AM",  # Wrong format
                birth_place="New Delhi",
                preffered_language="en"
            )
        assert "Invalid birth time" in str(exc_info.value)
    
    def test_invalid_language(self):
        """
        Test UserProfile with invalid language code.
        
        What: Validates that only "en" or "hi" are accepted.
        Why: Ensures language preference is limited to supported languages.
        Args: Invalid language codes (fr, es, etc.).
        """
        with pytest.raises(ValidationError):
            UserProfile(
                name="John Doe",
                birth_date="1990-01-15",
                birth_time="10:30",
                birth_place="New Delhi",
                preffered_language="fr"  # Invalid language
            )


class TestChatRequest:
    """
    Test ChatRequest model.
    
    Tests: Required fields, nested UserProfile validation.
    Why: ChatRequest is the main input for chat endpoints.
    Args: session_id, message, user_profile combinations.
    """
    
    def test_valid_chat_request(self, mock_user_profile):
        """
        Test creating a valid ChatRequest.
        
        What: Validates that a properly formatted ChatRequest can be created.
        Why: Ensures chat endpoint receives valid input.
        Args: Valid session_id, message, and UserProfile.
        """
        request = ChatRequest(
            session_id="session_123",
            message="What is my sun sign?",
            user_profile=mock_user_profile
        )
        assert request.session_id == "session_123"
        assert request.message == "What is my sun sign?"
        assert request.user_profile.name == mock_user_profile.name


class TestChatResponse:
    """
    Test ChatResponse model.
    
    Tests: Response structure, context_used list, zodiac_sign field.
    Why: ChatResponse defines the output format for chat endpoints.
    Args: response string, context_used list, zodiac_sign string.
    """
    
    def test_valid_chat_response(self):
        """
        Test creating a valid ChatResponse.
        
        What: Validates that a properly formatted ChatResponse can be created.
        Why: Ensures chat endpoint returns data in correct format.
        Args: Valid response string, context_used list, sun_sign, moon_sign, ascendant_sign, dasha_info strings.
        """
        response = ChatResponse(
            response="Your sun sign is Capricorn.",
            context_used=["zodiacs:Capricorn", "planetary_factors:Sun"],
            sun_sign="Capricorn",
            moon_sign="Leo",
            ascendant_sign="Aries",
            dasha_info="Ketu (2020-01-01 to 2027-01-01) - Current Bhukti: Ketu-Ketu (2020-01-01 to 2020-06-01)"
        )
        assert response.response == "Your sun sign is Capricorn."
        assert len(response.context_used) == 2
        assert response.sun_sign == "Capricorn"
        assert response.moon_sign == "Leo"
        assert response.ascendant_sign == "Aries"
        assert "Ketu" in response.dasha_info


class TestMetadataFilters:
    """
    Test MetadataFilters model.
    
    Tests: Optional fields, list constraints, valid enum values.
    Why: MetadataFilters control RAG query filtering in ChromaDB.
    Args: zodiacs, planetary_factors, life_areas, nakshtra lists.
    """
    
    def test_valid_metadata_filters(self):
        """
        Test creating valid MetadataFilters.
        
        What: Validates that MetadataFilters accepts valid filter values.
        Why: Ensures RAG queries use correct metadata filters.
        Args: Valid zodiac signs, planetary factors, life areas, nakshatras.
        """
        filters = MetadataFilters(
            zodiacs=["Aries", "Taurus"],
            planetary_factors=["Sun", "Moon"],
            life_areas=["love", "career"],
            nakshtra=["Ashwini", "Bharani"]
        )
        assert len(filters.zodiacs) == 2
        assert len(filters.planetary_factors) == 2
        assert len(filters.life_areas) == 2
        assert len(filters.nakshtra) == 2
    
    def test_empty_metadata_filters(self):
        """
        Test MetadataFilters with all None values.
        
        What: Validates that all fields can be None.
        Why: RAG queries may not need all filter types.
        Args: All None values.
        """
        filters = MetadataFilters()
        assert filters.zodiacs is None
        assert filters.planetary_factors is None
        assert filters.life_areas is None
        assert filters.nakshtra is None
    
    def test_invalid_zodiac_sign(self):
        """
        Test MetadataFilters with invalid zodiac sign.
        
        What: Validates that only valid zodiac signs are accepted.
        Why: Prevents invalid filter values from being used.
        Args: Invalid zodiac sign string.
        """
        with pytest.raises(ValidationError):
            MetadataFilters(zodiacs=["InvalidSign"])


class TestRAGQueryOutput:
    """
    Test RAGQueryOutput model.
    
    Tests: needs_rag boolean, optional fields, nested MetadataFilters.
    Why: RAGQueryOutput defines structured LLM output for RAG decisions.
    Args: needs_rag bool, metadata_filters, rag_query, reasoning strings.
    """
    
    def test_valid_rag_query_output_with_rag(self):
        """
        Test RAGQueryOutput when RAG is needed.
        
        What: Validates RAGQueryOutput with needs_rag=True and all fields populated.
        Why: Ensures structured output includes all necessary RAG information.
        Args: needs_rag=True, metadata_filters, rag_query, reasoning.
        """
        output = RAGQueryOutput(
            needs_rag=True,
            metadata_filters=MetadataFilters(zodiacs=["Aries"]),
            rag_query="What is the personality of Aries?",
            reasoning="User asked about personality traits"
        )
        assert output.needs_rag is True
        assert output.metadata_filters is not None
        assert output.rag_query == "What is the personality of Aries?"
        assert output.reasoning is not None
    
    def test_valid_rag_query_output_without_rag(self):
        """
        Test RAGQueryOutput when RAG is not needed.
        
        What: Validates RAGQueryOutput with needs_rag=False.
        Why: Ensures model can indicate when RAG is unnecessary.
        Args: needs_rag=False, all other fields None.
        """
        output = RAGQueryOutput(needs_rag=False)
        assert output.needs_rag is False
        assert output.metadata_filters is None
        assert output.rag_query is None
        assert output.reasoning is None


class TestKundaliDetails:
    """
    Test KundaliDetails and related models.
    
    Tests: Nested model structure, field types, optional fields.
    Why: KundaliDetails is the core astrological data structure.
    Args: Complete kundali data including planets, houses, aspects.
    """
    
    def test_valid_kundali_details(self, mock_kundali_details):
        """
        Test creating valid KundaliDetails.
        
        What: Validates that complete kundali data can be structured correctly.
        Why: Ensures astrological calculations produce valid output.
        Args: Complete kundali data with all required fields.
        """
        assert mock_kundali_details.user_name == "Test User"
        assert mock_kundali_details.key_positions.sun.sign == "Capricorn"
        assert mock_kundali_details.key_positions.moon.sign == "Leo"
        assert len(mock_kundali_details.planets) > 0
        assert len(mock_kundali_details.houses) > 0
    
    def test_planetary_position_optional_fields(self):
        """
        Test PlanetaryPosition with optional fields as None.
        
        What: Validates that optional fields can be None.
        Why: Some planetary positions may not have all details.
        Args: PlanetaryPosition with some fields as None.
        """
        position = PlanetaryPosition(
            sign="Aries",
            nakshatra=None,
            nakshatra_pada=None,
            nakshatra_lord=None,
            rasi_lord=None,
            sub_lord=None,
            sub_sub_lord=None,
            longitude=5.8
        )
        assert position.sign == "Aries"
        assert position.nakshatra is None
        assert position.longitude == 5.8

