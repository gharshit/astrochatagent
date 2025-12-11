"""
Tests for utility functions in app.utils.

This module tests:
- get_lat_lon() geocoding function
- get_utc_offset() timezone calculation
- parse_birth_datetime() date/time parsing
- fetch_kundali_details() complete kundali calculation flow
- safe_get_consolidated_chart_data() error handling

Why: Utils contain critical functions for kundali calculation and location services.
Args: Birth dates/times, place names, coordinates, VedicHoroscopeData instances.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from fastapi import HTTPException
from app.utils import (
    get_lat_lon,
    get_utc_offset,
    parse_birth_datetime,
    fetch_kundali_details,
    safe_get_consolidated_chart_data
)
from app.models import UserProfile


class TestGetLatLon:
    """
    Test get_lat_lon() geocoding function.
    
    Tests: Successful geocoding, location not found, error handling.
    Why: Geocoding is required to convert place names to coordinates for kundali calculation.
    Args: Place name strings, FastAPI Request with geocoder.
    """
    
    def test_get_lat_lon_success(self, mock_fastapi_request):
        """
        Test get_lat_lon() successfully geocodes a place.
        
        What: Validates that place name is converted to latitude/longitude.
        Why: Coordinates are required for accurate astrological calculations.
        Args: Valid place name, mock geocoder returning location.
        """
        lat, lon = get_lat_lon("New Delhi, India", mock_fastapi_request)
        
        assert lat == 28.6139
        assert lon == 77.2090
        mock_fastapi_request.app.state.geocoder.geocode.assert_called_once()
    
    def test_get_lat_lon_not_found(self, mock_fastapi_request):
        """
        Test get_lat_lon() raises HTTPException when location not found.
        
        What: Validates error handling for invalid place names.
        Why: Invalid locations should return proper error responses.
        Args: Invalid place name, geocoder returning None.
        """
        mock_fastapi_request.app.state.geocoder.geocode.return_value = None
        
        with pytest.raises(HTTPException) as exc_info:
            get_lat_lon("InvalidPlace12345", mock_fastapi_request)
        
        assert exc_info.value.status_code == 404
        assert "Location not found" in exc_info.value.detail
    
    def test_get_lat_lon_error(self, mock_fastapi_request):
        """
        Test get_lat_lon() handles geocoding errors.
        
        What: Validates error handling for geocoding exceptions.
        Why: External services may fail, should return proper error.
        Args: Geocoder raising exception.
        """
        mock_fastapi_request.app.state.geocoder.geocode.side_effect = Exception("Geocoding error")
        
        with pytest.raises(HTTPException) as exc_info:
            get_lat_lon("Test Place", mock_fastapi_request)
        
        assert exc_info.value.status_code == 500


class TestGetUtcOffset:
    """
    Test get_utc_offset() timezone calculation function.
    
    Tests: UTC offset calculation, timezone not found fallback, error handling.
    Why: UTC offset is required for accurate birth time conversion.
    Args: Latitude, longitude, birth date, birth time.
    """
    
    @patch('app.utils.TimezoneFinder')
    @patch('app.utils.datetime')
    def test_get_utc_offset_success(self, mock_datetime, mock_timezone_finder):
        """
        Test get_utc_offset() calculates correct UTC offset.
        
        What: Validates UTC offset calculation for given coordinates and datetime.
        Why: Accurate timezone conversion is critical for astrological calculations.
        Args: Latitude, longitude, birth date, birth time.
        """
        import sys
        import pytz as real_pytz
        
        mock_tf = Mock()
        mock_tf.timezone_at.return_value = "Asia/Kolkata"
        mock_timezone_finder.return_value = mock_tf
        
        # Mock datetime parsing
        mock_datetime.datetime.strptime.return_value = Mock(year=1990, month=1, day=15, hour=10, minute=30)
        
        # Mock pytz timezone - pytz is imported inside the function
        mock_tz = Mock()
        mock_dt = Mock()
        mock_offset = Mock()
        mock_offset.total_seconds.return_value = 19800  # +05:30
        mock_dt.utcoffset.return_value = mock_offset
        mock_tz.localize.return_value = mock_dt
        
        # Create a mock pytz module and patch sys.modules so the import inside the function uses it
        mock_pytz = Mock()
        mock_pytz.timezone.return_value = mock_tz
        
        # Patch sys.modules to return our mock when pytz is imported
        original_pytz = sys.modules.get('pytz')
        sys.modules['pytz'] = mock_pytz
        
        try:
            offset = get_utc_offset(28.6139, 77.2090, "1990-01-15", "10:30")
            assert offset == "+05:30"
        finally:
            # Restore original pytz
            if original_pytz is not None:
                sys.modules['pytz'] = original_pytz
            elif 'pytz' in sys.modules:
                del sys.modules['pytz']
    
    @patch('app.utils.TimezoneFinder')
    def test_get_utc_offset_timezone_not_found(self, mock_timezone_finder):
        """
        Test get_utc_offset() defaults to UTC when timezone not found.
        
        What: Validates fallback to UTC when timezone cannot be determined.
        Why: Ensures function always returns valid offset even on failure.
        Args: Coordinates without timezone data.
        """
        mock_tf = Mock()
        mock_tf.timezone_at.return_value = None
        mock_timezone_finder.return_value = mock_tf
        
        offset = get_utc_offset(0.0, 0.0, "1990-01-15", "10:30")
        
        assert offset == "+00:00"
    
    @patch('app.utils.TimezoneFinder')
    def test_get_utc_offset_error_handling(self, mock_timezone_finder):
        """
        Test get_utc_offset() handles errors gracefully.
        
        What: Validates error handling returns UTC fallback.
        Why: Ensures function never fails completely.
        Args: Exception during timezone calculation.
        """
        mock_tf = Mock()
        mock_tf.timezone_at.side_effect = Exception("Timezone error")
        mock_timezone_finder.return_value = mock_tf
        
        offset = get_utc_offset(28.6139, 77.2090, "1990-01-15", "10:30")
        
        assert offset == "+00:00"


class TestParseBirthDatetime:
    """
    Test parse_birth_datetime() date/time parsing function.
    
    Tests: Valid date/time parsing, invalid format handling.
    Why: Birth datetime must be parsed into components for VedicHoroscopeData.
    Args: Birth date (YYYY-MM-DD), birth time (HH:MM) strings.
    """
    
    def test_parse_birth_datetime_success(self):
        """
        Test parse_birth_datetime() successfully parses valid date/time.
        
        What: Validates that date/time strings are parsed into components.
        Why: VedicHoroscopeData requires separate year, month, day, hour, minute.
        Args: Valid date (YYYY-MM-DD) and time (HH:MM) strings.
        """
        year, month, day, hour, minute = parse_birth_datetime("1990-01-15", "10:30")
        
        assert year == 1990
        assert month == 1
        assert day == 15
        assert hour == 10
        assert minute == 30
    
    def test_parse_birth_datetime_invalid_date(self):
        """
        Test parse_birth_datetime() raises HTTPException for invalid date format.
        
        What: Validates error handling for invalid date formats.
        Why: Invalid dates should be caught early with proper error messages.
        Args: Invalid date format string.
        """
        with pytest.raises(HTTPException) as exc_info:
            parse_birth_datetime("15-01-1990", "10:30")
        
        assert exc_info.value.status_code == 400
        assert "Invalid date" in exc_info.value.detail
    
    def test_parse_birth_datetime_invalid_time(self):
        """
        Test parse_birth_datetime() raises HTTPException for invalid time format.
        
        What: Validates error handling for invalid time formats.
        Why: Invalid times should be caught early with proper error messages.
        Args: Invalid time format string.
        """
        with pytest.raises(HTTPException) as exc_info:
            parse_birth_datetime("1990-01-15", "10:30 AM")
        
        assert exc_info.value.status_code == 400
        assert "Invalid" in exc_info.value.detail


class TestSafeGetConsolidatedChartData:
    """
    Test safe_get_consolidated_chart_data() error handling function.
    
    Tests: Successful consolidation, fallback methods, error handling.
    Why: Handles polars version compatibility issues in vedicastro library.
    Args: VedicHoroscopeData instance, planets_data, houses_data.
    """
    
    def test_safe_get_consolidated_chart_data_success(self, mock_vedic_data):
        """
        Test safe_get_consolidated_chart_data() successfully consolidates chart data.
        
        What: Validates that chart data is consolidated successfully.
        Why: Consolidated chart provides grouped astrological data.
        Args: Valid VedicHoroscopeData with planets and houses data.
        """
        mock_vedic_data.get_consolidated_chart_data.return_value = {"test": "data"}
        planets_data = []
        houses_data = []
        
        result = safe_get_consolidated_chart_data(mock_vedic_data, planets_data, houses_data)
        
        assert result == {"test": "data"}
        mock_vedic_data.get_consolidated_chart_data.assert_called_once()
    
    def test_safe_get_consolidated_chart_data_fallback(self, mock_vedic_data):
        """
        Test safe_get_consolidated_chart_data() uses fallback on error.
        
        What: Validates that function tries alternative methods on failure.
        Why: Handles polars compatibility issues gracefully.
        Args: VedicHoroscopeData raising TypeError on first attempt.
        """
        mock_vedic_data.get_consolidated_chart_data.side_effect = [
            TypeError("Polars error"),
            {"fallback": "data"}
        ]
        planets_data = []
        houses_data = []
        
        result = safe_get_consolidated_chart_data(mock_vedic_data, planets_data, houses_data)
        
        assert result == {"fallback": "data"}
        assert mock_vedic_data.get_consolidated_chart_data.call_count == 2


class TestFetchKundaliDetails:
    """
    Test fetch_kundali_details() complete kundali calculation function.
    
    Tests: Complete kundali calculation flow, error handling, data conversion.
    Why: Main function that orchestrates entire kundali calculation process.
    Args: UserProfile, FastAPI Request with geocoder.
    """
    
    @pytest.mark.asyncio
    @patch('app.utils.get_lat_lon')
    @patch('app.utils.parse_birth_datetime')
    @patch('app.utils.get_utc_offset')
    @patch('app.utils.VedicHoroscopeData')
    @patch('app.utils.safe_get_consolidated_chart_data')
    async def test_fetch_kundali_details_success(
        self,
        mock_safe_consolidate,
        mock_vedic_class,
        mock_get_utc_offset,
        mock_parse_datetime,
        mock_get_lat_lon,
        mock_user_profile,
        mock_fastapi_request
    ):
        """
        Test fetch_kundali_details() successfully calculates kundali.
        
        What: Validates complete kundali calculation flow from user profile to KundaliDetails.
        Why: Ensures end-to-end kundali generation works correctly.
        Args: Valid UserProfile, mock geocoder, mock VedicHoroscopeData.
        """
        # Setup mocks
        mock_get_lat_lon.return_value = (28.6139, 77.2090)
        mock_parse_datetime.return_value = (1990, 1, 15, 10, 30)
        mock_get_utc_offset.return_value = "+05:30"
        
        mock_vedic_instance = mock_vedic_class.return_value
        mock_vedic_instance.generate_chart.return_value = {
            "Sun": Mock(sign="Capricorn", lon=285.5),
            "Moon": Mock(sign="Leo", lon=135.2),
            "Asc": Mock(sign="Aries", lon=5.8)
        }
        mock_vedic_instance.get_planets_data_from_chart.return_value = []
        mock_vedic_instance.get_houses_data_from_chart.return_value = []
        mock_vedic_instance.get_planetary_aspects.return_value = []
        mock_vedic_instance.compute_vimshottari_dasa.return_value = {}
        mock_vedic_instance.get_rl_nl_sl_data.return_value = {
            "Nakshatra": "Uttara Ashadha",
            "Pada": 1,
            "NakshatraLord": "Sun",
            "RasiLord": "Saturn",
            "SubLord": "Sun",
            "SubSubLord": "Sun"
        }
        mock_vedic_instance.ayanamsa = "Lahiri"
        mock_vedic_instance.house_system = "Equal"
        mock_safe_consolidate.return_value = None
        
        result = await fetch_kundali_details(mock_user_profile, mock_fastapi_request)
        
        assert result is not None
        assert result.user_name == mock_user_profile.name
        assert result.key_positions.sun.sign == "Capricorn"
        mock_get_lat_lon.assert_called_once()
        mock_parse_datetime.assert_called_once()
        mock_get_utc_offset.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_fetch_kundali_details_geocoding_error(self, mock_user_profile, mock_fastapi_request):
        """
        Test fetch_kundali_details() handles geocoding errors.
        
        What: Validates error handling when geocoding fails.
        Why: Geocoding errors should propagate with proper HTTP status.
        Args: UserProfile with invalid place, geocoder raising HTTPException.
        """
        from app.utils import get_lat_lon
        with patch('app.utils.get_lat_lon') as mock_get_lat_lon:
            mock_get_lat_lon.side_effect = HTTPException(status_code=404, detail="Location not found")
            
            with pytest.raises(HTTPException) as exc_info:
                await fetch_kundali_details(mock_user_profile, mock_fastapi_request)
            
            assert exc_info.value.status_code == 404

