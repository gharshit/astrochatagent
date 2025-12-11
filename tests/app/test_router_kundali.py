"""
Tests for kundali router endpoints in app.router.kundali_router.

This module tests:
- POST /v1/kundali/ endpoint
- Kundali generation from user profile
- Error handling and HTTP status codes
- Response model validation

Why: Kundali router provides astrological chart generation endpoint.
Args: UserProfile with birth details, FastAPI Request with geocoder.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi import HTTPException, status
from app.router.kundali_router import generate_kundali
from app.models import UserProfile, KundaliDetails


class TestGenerateKundaliEndpoint:
    """
    Test POST /v1/kundali/ endpoint.
    
    Tests: Successful kundali generation, error handling, validation.
    Why: Main endpoint for generating astrological charts.
    Args: UserProfile, FastAPI Request with geocoder.
    """
    
    @pytest.mark.asyncio
    async def test_generate_kundali_success(self, mock_user_profile, mock_fastapi_request, mock_kundali_details):
        """
        Test generate_kundali endpoint successfully generates kundali.
        
        What: Validates that kundali is generated from user profile.
        Why: Ensures core functionality works correctly.
        Args: Valid UserProfile, mock geocoder, mock VedicHoroscopeData.
        """
        with patch('app.router.kundali_router.fetch_kundali_details', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = mock_kundali_details
            
            result = await generate_kundali(mock_user_profile, mock_fastapi_request)
            
            assert isinstance(result, KundaliDetails)
            assert result.user_name == mock_user_profile.name
            assert result.key_positions.sun.sign == "Capricorn"
            mock_fetch.assert_called_once_with(mock_user_profile, mock_fastapi_request)
    
    @pytest.mark.asyncio
    async def test_generate_kundali_geocoding_error(self, mock_user_profile, mock_fastapi_request):
        """
        Test generate_kundali endpoint handles geocoding errors.
        
        What: Validates error handling when place cannot be geocoded.
        Why: Ensures proper error response for invalid locations.
        Args: UserProfile with invalid birth_place.
        """
        with patch('app.router.kundali_router.fetch_kundali_details', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.side_effect = HTTPException(status_code=404, detail="Location not found")
            
            with pytest.raises(HTTPException) as exc_info:
                await generate_kundali(mock_user_profile, mock_fastapi_request)
            
            assert exc_info.value.status_code == 404
            assert "Location not found" in exc_info.value.detail
    
    @pytest.mark.asyncio
    async def test_generate_kundali_validation_error(self, mock_user_profile, mock_fastapi_request):
        """
        Test generate_kundali endpoint handles validation errors.
        
        What: Validates error handling when birth date/time format is invalid.
        Why: Ensures proper error response for invalid input.
        Args: UserProfile with invalid birth_date or birth_time.
        """
        with patch('app.router.kundali_router.fetch_kundali_details', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.side_effect = ValueError("Invalid birth date format")
            
            with pytest.raises(HTTPException) as exc_info:
                await generate_kundali(mock_user_profile, mock_fastapi_request)
            
            assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
            assert "Invalid input data" in exc_info.value.detail
    
    @pytest.mark.asyncio
    async def test_generate_kundali_calculation_error(self, mock_user_profile, mock_fastapi_request):
        """
        Test generate_kundali endpoint handles calculation errors.
        
        What: Validates error handling when kundali calculation fails.
        Why: Ensures proper error response for internal errors.
        Args: UserProfile causing calculation exception.
        """
        with patch('app.router.kundali_router.fetch_kundali_details', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.side_effect = Exception("Calculation error")
            
            with pytest.raises(HTTPException) as exc_info:
                await generate_kundali(mock_user_profile, mock_fastapi_request)
            
            assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert "Error generating kundali" in exc_info.value.detail
    
    @pytest.mark.asyncio
    async def test_generate_kundali_http_exception_passthrough(self, mock_user_profile, mock_fastapi_request):
        """
        Test generate_kundali endpoint passes through HTTPExceptions.
        
        What: Validates that HTTPExceptions are re-raised with original status codes.
        Why: Ensures proper error propagation.
        Args: UserProfile causing HTTPException.
        """
        with patch('app.router.kundali_router.fetch_kundali_details', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.side_effect = HTTPException(status_code=400, detail="Custom error")
            
            with pytest.raises(HTTPException) as exc_info:
                await generate_kundali(mock_user_profile, mock_fastapi_request)
            
            assert exc_info.value.status_code == 400
            assert exc_info.value.detail == "Custom error"

