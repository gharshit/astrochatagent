"""
Utility functions for kundali calculations and location services.
"""

import datetime
from typing import Tuple, List, Dict, Any
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
    PlanetaryAspect,
    DasaDetails,
    BhuktiDetails
)
from fastapi import Request, HTTPException
from vedicastro.VedicAstro import VedicHoroscopeData
from timezonefinder import TimezoneFinder
from helper.utils.logger import setup_logger
import polars as pl
import collections

# Setup logger for utils
logger = setup_logger(name="app.utils", level=20)  # INFO level


def safe_get_consolidated_chart_data(vedic_data, planets_data, houses_data):
    """
    Safely get consolidated chart data, handling polars version compatibility issues.
    
    This is a workaround for a bug in vedicastro where boolean columns cause
    issues with map_elements(list, ...) in certain polars versions.
    
    Args:
        vedic_data: VedicHoroscopeData instance
        planets_data: Planets data namedtuple collection
        houses_data: Houses data namedtuple collection
        
    Returns:
        Consolidated chart data or None if extraction fails
    """
    try:
        # Try the standard method first
        return vedic_data.get_consolidated_chart_data(
            planets_data=planets_data,
            houses_data=houses_data,
            return_style="dataframe_records"
        )
    except (TypeError, AttributeError) as e:
        logger.debug(f"Standard consolidation failed: {e}, trying alternative method...")
        try:
            # Try without dataframe_records style
            return vedic_data.get_consolidated_chart_data(
                planets_data=planets_data,
                houses_data=houses_data,
                return_style=None
            )
        except Exception as e2:
            logger.debug(f"Alternative consolidation also failed: {e2}, creating manual consolidation...")
            # Manual fallback: create a simple grouped structure
            try:
                req_cols = ["Rasi", "Object", "isRetroGrade", "LonDecDeg", "SignLonDMS", "SignLonDecDeg"]
                planets_df = pl.DataFrame(planets_data).select(req_cols)
                houses_df = pl.DataFrame(houses_data).with_columns(
                    pl.lit(False).alias("isRetroGrade")
                ).select(req_cols)
                
                df_concat = pl.concat([houses_df, planets_df])
                
                # Group by Rasi manually
                result = {}
                for row in df_concat.iter_rows(named=True):
                    rasi = row['Rasi']
                    if rasi not in result:
                        result[rasi] = {
                            'Object': [],
                            'isRetroGrade': [],
                            'LonDecDeg': [],
                            'SignLonDMS': [],
                            'SignLonDecDeg': []
                        }
                    result[rasi]['Object'].append(row['Object'])
                    result[rasi]['isRetroGrade'].append(row['isRetroGrade'])
                    result[rasi]['LonDecDeg'].append(row['LonDecDeg'])
                    result[rasi]['SignLonDMS'].append(row['SignLonDMS'])
                    result[rasi]['SignLonDecDeg'].append(row['SignLonDecDeg'])
                
                # Convert to list of dicts format
                return [{"Rasi": k, **v} for k, v in result.items()]
            except Exception as e3:
                logger.warning(f"Manual consolidation also failed: {e3}")
                return None


def get_lat_lon(place: str, request: Request) -> Tuple[float, float]:
    """
    Get the latitude and longitude of a place.
    
    Args:
        place: Birth place name
        request: FastAPI request object
        
    Returns:
        Tuple of (latitude, longitude)
    """
    logger.info(f"Getting coordinates for place: {place}")
    try:
        geolocator = request.app.state.geocoder
        logger.debug("Geocoder retrieved from app state")
        
        location = geolocator.geocode(place, addressdetails=True)
        if not location:
            logger.warning(f"Location not found: {place}")
            raise HTTPException(status_code=404, detail="Location not found")
        
        lat, lon = location.latitude, location.longitude
        logger.info(f"✓ Found coordinates - Latitude: {lat}, Longitude: {lon}")
        return lat, lon
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting coordinates for {place}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting latitude and longitude: {e}")


def get_utc_offset(latitude: float, longitude: float, birth_date: str, birth_time: str) -> str:
    """
    Get UTC offset for a given location and birth datetime.
    
    Args:
        latitude: Latitude of the location
        longitude: Longitude of the location
        birth_date: Birth date in YYYY-MM-DD format
        birth_time: Birth time in HH:MM format
        
    Returns:
        UTC offset string (e.g., "+05:30" or "-05:30")
    """
    logger.debug(f"Calculating UTC offset for lat: {latitude}, lon: {longitude}, "
                f"date: {birth_date}, time: {birth_time}")
    try:
        import pytz
        tf = TimezoneFinder()
        timezone_str = tf.timezone_at(lat=latitude, lng=longitude)
        
        if not timezone_str:
            logger.warning(f"Timezone not found for coordinates, defaulting to UTC")
            return "+00:00"
        
        logger.debug(f"Found timezone: {timezone_str}")
        
        # Parse birth datetime
        birth_datetime_str = f"{birth_date} {birth_time}:00"
        birth_datetime = datetime.datetime.strptime(birth_datetime_str, "%Y-%m-%d %H:%M:%S")
        
        # Get timezone info and calculate offset
        tz = pytz.timezone(timezone_str)
        local_dt = tz.localize(birth_datetime)
        utc_offset_seconds = local_dt.utcoffset().total_seconds()
        
        # Convert seconds to hours and minutes
        hours = int(utc_offset_seconds // 3600)
        minutes = int((utc_offset_seconds % 3600) // 60)
        
        # Format as +HH:MM or -HH:MM
        sign = "+" if hours >= 0 else "-"
        utc_offset = f"{sign}{abs(hours):02d}:{abs(minutes):02d}"
        logger.info(f"✓ UTC offset calculated: {utc_offset}")
        return utc_offset
    except Exception as e:
        logger.warning(f"Error calculating UTC offset, defaulting to UTC: {str(e)}")
        return "+00:00"


def parse_birth_datetime(birth_date: str, birth_time: str) -> Tuple[int, int, int, int, int]:
    """
    Parse birth date and time into separate components.
    
    Args:
        birth_date: Birth date in YYYY-MM-DD format
        birth_time: Birth time in HH:MM format
        
    Returns:
        Tuple of (year, month, day, hour, minute)
    """
    logger.debug(f"Parsing birth datetime - Date: {birth_date}, Time: {birth_time}")
    try:
        # Parse date
        date_obj = datetime.datetime.strptime(birth_date, "%Y-%m-%d")
        year = date_obj.year
        month = date_obj.month
        day = date_obj.day
        
        # Parse time
        time_obj = datetime.datetime.strptime(birth_time, "%H:%M")
        hour = time_obj.hour
        minute = time_obj.minute
        
        logger.debug(f"Parsed datetime - Year: {year}, Month: {month}, Day: {day}, "
                    f"Hour: {hour}, Minute: {minute}")
        return year, month, day, hour, minute
    except ValueError as e:
        logger.error(f"Invalid date/time format - Date: {birth_date}, Time: {birth_time}, Error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid date or time format: {e}")


async def fetch_kundali_details(user_profile: UserProfile, request: Request) -> KundaliDetails:
    """
    Fetch kundali details using VedicHoroscopeData.
    
    Extracts birth information from user profile and calculates kundali
    using VedicHoroscopeData class.
    
    Args:
        user_profile: User profile with birth details
        request: FastAPI request object
        
    Returns:
        KundaliDetails: Pydantic model containing all kundali details
        
    Raises:
        HTTPException: If there's an error processing the kundali
    """
    logger.info("=" * 60)
    logger.info("Starting kundali calculation process")
    logger.info(f"User: {user_profile.name}")
    logger.info("=" * 60)
    
    try:
        # Step 1: Get latitude and longitude from birth place
        logger.info("Step 1: Getting coordinates for birth place...")
        latitude, longitude = get_lat_lon(user_profile.birth_place, request)
        
        # Step 2: Parse birth date and time
        logger.info("Step 2: Parsing birth date and time...")
        year, month, day, hour, minute = parse_birth_datetime(
            user_profile.birth_date,
            user_profile.birth_time
        )
        
        # Step 3: Get UTC offset for the location
        logger.info("Step 3: Calculating UTC offset...")
        utc_offset = get_utc_offset(
            latitude,
            longitude,
            user_profile.birth_date,
            user_profile.birth_time
        )
        
        # Step 4: Create VedicHoroscopeData instance
        logger.info("Step 4: Creating VedicHoroscopeData instance...")
        logger.debug(f"Parameters - Year: {year}, Month: {month}, Day: {day}, "
                    f"Hour: {hour}, Minute: {minute}, UTC: {utc_offset}, "
                    f"Lat: {latitude}, Lon: {longitude}")
        
        vedic_data = VedicHoroscopeData(
            year=year,
            month=month,
            day=day,
            hour=hour,
            minute=minute,
            second=0,  # Default to 0 seconds
            utc=utc_offset,
            latitude=latitude,
            longitude=longitude,
            ayanamsa="Lahiri",  # Default ayanamsa
            house_system="Equal"  # Default house system
        )
        logger.info("✓ VedicHoroscopeData instance created")
        
        # Step 5: Generate chart
        logger.info("Step 5: Generating astrological chart...")
        chart = vedic_data.generate_chart()
        logger.info("✓ Chart generated successfully")
        
        # Step 6: Extract planets data
        logger.info("Step 6: Extracting planets data...")
        planets_data = vedic_data.get_planets_data_from_chart(chart)
        logger.info(f"✓ Extracted data for {len(planets_data)} planetary objects")
        
        # Step 7: Extract houses data
        logger.info("Step 7: Extracting houses data...")
        houses_data = vedic_data.get_houses_data_from_chart(chart)
        logger.info(f"✓ Extracted data for {len(houses_data)} houses")
        
        # Step 8: Extract planetary aspects
        logger.info("Step 8: Calculating planetary aspects...")
        planetary_aspects = vedic_data.get_planetary_aspects(chart)
        logger.info(f"✓ Found {len(planetary_aspects)} planetary aspects")
        
        # Step 9: Extract consolidated chart data (with error handling for polars compatibility)
        logger.info("Step 9: Consolidating chart data...")
        consolidated_data = safe_get_consolidated_chart_data(
            vedic_data   = vedic_data,
            planets_data = planets_data,
            houses_data  = houses_data
        )
        if consolidated_data:
            logger.info("✓ Chart data consolidated")
        else:
            logger.warning("Chart consolidation skipped (optional data)")
        
        # Step 10: Extract Vimshottari Dasa
        logger.info("Step 10: Computing Vimshottari Dasa...")
        vimshottari_dasa = vedic_data.compute_vimshottari_dasa(chart)
        logger.info("✓ Vimshottari Dasa calculated")
        
          # Step 11: Extract key planetary positions
        logger.info("Step 11: Extracting key planetary positions (Sun, Moon, Ascendant)...")
        sun       = chart.get("Sun")
        moon      = chart.get("Moon")
        ascendant = chart.get("Asc")
        
        # Extract sun sign, moon sign, ascendant sign
        sun_sign       = sun.sign if sun else None
        moon_sign      = moon.sign if moon else None
        ascendant_sign = ascendant.sign if ascendant else None
        
          # Extract nakshatra details for sun, moon, and ascendant
        sun_nakshatra_data       = vedic_data.get_rl_nl_sl_data(deg=sun.lon) if sun else {}
        moon_nakshatra_data      = vedic_data.get_rl_nl_sl_data(deg=moon.lon) if moon else {}
        ascendant_nakshatra_data = vedic_data.get_rl_nl_sl_data(deg=ascendant.lon) if ascendant else {}
        
        # Extract lagna lord (ascendant lord)
        lagna_lord = ascendant_nakshatra_data.get("RasiLord") if ascendant_nakshatra_data else None
        
        logger.info(f"✓ Key positions - Sun: {sun_sign}, Moon: {moon_sign}, "
                   f"Ascendant: {ascendant_sign}, Lagna Lord: {lagna_lord}")
        
        # Step 12: Convert planets_data and houses_data to Pydantic models
        logger.info("Step 12: Converting data structures to Pydantic models...")
        planets_list: List[PlanetData] = []
        for planet in planets_data:
            planets_list.append(PlanetData(
                object            = planet.Object,
                rasi              = planet.Rasi,
                is_retrograde     = planet.isRetroGrade,
                longitude_dec_deg = planet.LonDecDeg,
                sign_lon_dms      = planet.SignLonDMS,
                sign_lon_dec_deg  = planet.SignLonDecDeg,
                lat_dms           = planet.LatDMS,
                nakshatra         = planet.Nakshatra,
                rasi_lord         = planet.RasiLord,
                nakshatra_lord    = planet.NakshatraLord,
                sub_lord          = planet.SubLord,
                sub_sub_lord      = planet.SubSubLord,
                house_nr          = planet.HouseNr
            ))
        
        houses_list: List[HouseData] = []
        for house in houses_data:
            houses_list.append(HouseData(
                object            = house.Object,
                house_nr          = house.HouseNr,
                rasi              = house.Rasi,
                longitude_dec_deg = house.LonDecDeg,
                sign_lon_dms      = house.SignLonDMS,
                sign_lon_dec_deg  = house.SignLonDecDeg,
                deg_size          = house.DegSize,
                nakshatra         = house.Nakshatra,
                rasi_lord         = house.RasiLord,
                nakshatra_lord    = house.NakshatraLord,
                sub_lord          = house.SubLord,
                sub_sub_lord      = house.SubSubLord
            ))
        
        logger.info(f"✓ Converted {len(planets_list)} planets and {len(houses_list)} houses to Pydantic models")
        
        # Step 13: Convert planetary aspects to Pydantic models
        logger.info("Step 13: Converting planetary aspects to Pydantic models...")
        aspects_list: List[PlanetaryAspect] = []
        for aspect in planetary_aspects:
            aspects_list.append(PlanetaryAspect(
                P1         = aspect.get("P1", ""),
                P2         = aspect.get("P2", ""),
                AspectType = aspect.get("AspectType", ""),
                AspectDeg  = aspect.get("AspectDeg", 0),
                AspectOrb  = aspect.get("AspectOrb", 0.0)
            ))
        
        # Step 14: Convert Vimshottari Dasa to Pydantic models
        logger.info("Step 14: Converting Vimshottari Dasa to Pydantic models...")
        dasa_dict: Dict[str, DasaDetails] = {}
        for dasa_name, dasa_info in vimshottari_dasa.items():
            bhuktis_dict: Dict[str, BhuktiDetails] = {}
            for bhukti_name, bhukti_info in dasa_info.get("bhuktis", {}).items():
                bhuktis_dict[bhukti_name] = BhuktiDetails(
                    start = bhukti_info.get("start", ""),
                    end   = bhukti_info.get("end", "")
                )
            
            dasa_dict[dasa_name] = DasaDetails(
                start=dasa_info.get("start", ""),
                end=dasa_info.get("end", ""),
                bhuktis=bhuktis_dict
            )
        
        # Step 15: Build comprehensive kundali details using Pydantic models
        logger.info("Step 15: Building final kundali details Pydantic model...")
        kundali_details = KundaliDetails(
            user_name=user_profile.name,
            birth_details=BirthDetails(
                birth_date  = user_profile.birth_date,
                birth_time  = user_profile.birth_time,
                birth_place = user_profile.birth_place,
                year        = year,
                month       = month,
                day         = day,
                hour        = hour,
                minute      = minute,
                second      = 0
            ),
            location=LocationDetails(
                latitude   = latitude,
                longitude  = longitude,
                utc_offset = utc_offset
            ),
            chart_settings=ChartSettings(
                ayanamsa     = vedic_data.ayanamsa,
                house_system = vedic_data.house_system
            ),
            key_positions=KeyPositions(
                sun=PlanetaryPosition(
                    sign           = sun_sign,
                    nakshatra      = sun_nakshatra_data.get("Nakshatra"),
                    nakshatra_pada = sun_nakshatra_data.get("Pada"),
                    nakshatra_lord = sun_nakshatra_data.get("NakshatraLord"),
                    rasi_lord      = sun_nakshatra_data.get("RasiLord"),
                    sub_lord       = sun_nakshatra_data.get("SubLord"),
                    sub_sub_lord   = sun_nakshatra_data.get("SubSubLord"),
                    longitude      = round(sun.lon, 3) if sun else None
                ),
                moon=PlanetaryPosition(
                    sign           = moon_sign,
                    nakshatra      = moon_nakshatra_data.get("Nakshatra"),
                    nakshatra_pada = moon_nakshatra_data.get("Pada"),
                    nakshatra_lord = moon_nakshatra_data.get("NakshatraLord"),
                    rasi_lord      = moon_nakshatra_data.get("RasiLord"),
                    sub_lord       = moon_nakshatra_data.get("SubLord"),
                    sub_sub_lord   = moon_nakshatra_data.get("SubSubLord"),
                    longitude      = round(moon.lon, 3) if moon else None
                ),
                ascendant=PlanetaryPosition(
                    sign           = ascendant_sign,
                    nakshatra      = ascendant_nakshatra_data.get("Nakshatra"),
                    nakshatra_pada = ascendant_nakshatra_data.get("Pada"),
                    nakshatra_lord = ascendant_nakshatra_data.get("NakshatraLord"),
                    rasi_lord      = lagna_lord,
                    sub_lord       = ascendant_nakshatra_data.get("SubLord"),
                    sub_sub_lord   = ascendant_nakshatra_data.get("SubSubLord"),
                    longitude      = round(ascendant.lon, 3) if ascendant else None
                ),
                lagna_lord=lagna_lord
            ),
            planets            = planets_list,
            houses             = houses_list,
            planetary_aspects  = aspects_list,
            consolidated_chart = consolidated_data,
            vimshottari_dasa   = dasa_dict
        )
        
        logger.info("✓ Kundali details Pydantic model built successfully")
        logger.info("=" * 60)
        logger.info("Kundali calculation completed successfully")
        logger.info("=" * 60)
        
        return kundali_details
        
    except HTTPException as e:
        logger.error(f"HTTP error in kundali calculation: {e.status_code} - {e.detail}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in kundali calculation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching kundali details: {str(e)}"
        )