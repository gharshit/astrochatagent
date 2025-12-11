"""
Kundali router for generating kundali/horoscope details from user information.

This router provides endpoints for generating complete astrological charts
and kundali details based on user birth information.
"""

from fastapi import APIRouter, Request, HTTPException, status
from app.models import UserProfile, KundaliDetails
from app.utils import fetch_kundali_details
from helper.utils.logger import setup_logger

# Setup logger for kundali router
logger = setup_logger(name="app.router.kundali", level=20)  # INFO level


kundali_router = APIRouter(
    prefix="/v1/kundali",
    tags=["kundali"]
)


@kundali_router.post(
    "/",
    response_model=KundaliDetails,
    status_code=status.HTTP_200_OK,
    description="Generate kundali/horoscope details from user birth information",
    responses={
        200: {
            "description": "Kundali details generated successfully",
            "model": KundaliDetails
        },
        400: {
            "description": "Invalid input data (birth date, time, or place format)"
        },
        404: {
            "description": "Birth place not found (geocoding failed)"
        },
        500: {
            "description": "Internal server error during kundali calculation"
        }
    }
)
async def generate_kundali(user_profile: UserProfile, request: Request) -> KundaliDetails:
    """
    Generate complete kundali/horoscope details from user birth information.
    
    This endpoint calculates comprehensive astrological data including:
    - Planetary positions (Sun, Moon, all planets)
    - House positions (all 12 houses)
    - Nakshatra details (star, pada, lords)
    - Planetary aspects (conjunctions, oppositions, etc.)
    - Vimshottari Dasa periods (Maha Dasa and Bhukti)
    - Key astrological positions (Sun sign, Moon sign, Ascendant, Lagna Lord)
    
    Args:
        user_profile: User profile containing birth details
            - name (str): User's name
            - birth_date (str): Birth date in YYYY-MM-DD format
            - birth_time (str): Birth time in HH:MM format (24-hour)
            - birth_place (str): Birth place name (city, country)
            - preferred_language (str): Preferred language ("en" or "hi")
        request: FastAPI request object for accessing app state (geocoder)
        
    Returns:
        KundaliDetails: Complete kundali details Pydantic model containing:
            - user_name: User's name
            - birth_details: Parsed birth information
            - location: Geographic coordinates and UTC offset
            - chart_settings: Ayanamsa and house system used
            - key_positions: Sun, Moon, Ascendant positions
            - planets: List of all planetary positions
            - houses: List of all house positions
            - planetary_aspects: List of planetary aspects
            - consolidated_chart: Chart data grouped by Rasi
            - vimshottari_dasa: Dasa periods and timings
            
    Raises:
        HTTPException:
            - 400: If birth date or time format is invalid
            - 404: If birth place cannot be geocoded
            - 500: If there's an error during kundali calculation
    """
    logger.info("=" * 60)
    logger.info("Received kundali generation request")
    logger.info(f"User: {user_profile.name}")
    logger.info(f"Birth Date: {user_profile.birth_date}")
    logger.info(f"Birth Time: {user_profile.birth_time}")
    logger.info(f"Birth Place: {user_profile.birth_place}")
    logger.info("=" * 60)
    
    try:
        # Fetch kundali details
        logger.info("\n Generating kundali details...")
        kundali_details: KundaliDetails = await fetch_kundali_details(user_profile, request)
        
        logger.info("✓ Kundali generated successfully \n")
        logger.info(f"Sun Sign: {kundali_details.key_positions.sun.sign or 'N/A'}")
        logger.info(f"Moon Sign: {kundali_details.key_positions.moon.sign or 'N/A'}")
        logger.info(f"Ascendant: {kundali_details.key_positions.ascendant.sign or 'N/A'}")
        logger.info(f"Lagna Lord: {kundali_details.key_positions.lagna_lord or 'N/A'}")
        logger.info("=" * 60)
        
        return kundali_details
        
    except HTTPException as e:
        # Re-raise HTTP exceptions with their original status codes
        logger.error(f"HTTP error: {e.status_code} - {e.detail}")
        raise
    except ValueError as e:
        # Handle validation errors
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input data: {str(e)}"
        )
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Unexpected error generating kundali: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating kundali: {str(e)}"
        )

