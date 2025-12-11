


from pydantic import BaseModel, Field, field_validator
from typing import Literal, List, Optional
from datetime import datetime



class UserProfile(BaseModel):
    """
    User profile information
    """
    name               : str                = Field(..., description="User name")
    birth_date         : str                = Field(..., description="User birth date in YYYY-MM-DD format")
    birth_time         : str                = Field(..., description="User birth time in HH:MM format")
    birth_place        : str                = Field(..., description="User birth place")
    preffered_language : Literal["en","hi"] = Field(..., description="User preffered language from engish or hindi")
    
    # validate birth data and time
    @field_validator("birth_date")
    def validate_birth_date(cls, v):
        """
        Validate birth data
        """
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Invalid birth date")
        
        return v
    
    @field_validator("birth_time")
    def validate_birth_time(cls, v):
        """
        Validate birth time
        """
        try:
            datetime.strptime(v, "%H:%M")
        except ValueError:
            raise ValueError("Invalid birth time")
        return v
        

class ChatRequest(BaseModel):
    """
    
    """
    session_id   : str        = Field(..., description="The session ID for the chat")
    message      : str        = Field(..., description="User message")
    user_profile: UserProfile = Field(..., description="User profile")
    



class ChatResponse(BaseModel):
    """
    Response to the user message
    """
    response    : str       = Field(..., description="Response to the user message")
    context_used: List[str] = Field(..., description="List of context used to generate the response",examples=["career_transits","leo_traits"])
    zodiac_sign : str       = Field(..., description="Zodiac sign of the user")


# Kundali Details Models
class BirthDetails(BaseModel):
    """Birth details information."""
    birth_date : str = Field(..., description="Birth date in YYYY-MM-DD format")
    birth_time : str = Field(..., description="Birth time in HH:MM format")
    birth_place: str = Field(..., description="Birth place")
    year       : int = Field(..., description="Birth year")
    month      : int = Field(..., description="Birth month")
    day        : int = Field(..., description="Birth day")
    hour       : int = Field(..., description="Birth hour")
    minute     : int = Field(..., description="Birth minute")
    second     : int = Field(default=0, description="Birth second")


class LocationDetails(BaseModel):
    """Location details for birth place."""
    latitude  : float = Field(..., description="Latitude")
    longitude : float = Field(..., description="Longitude")
    utc_offset: str   = Field(..., description="UTC offset (e.g., +05:30)")


class ChartSettings(BaseModel):
    """Chart calculation settings."""
    ayanamsa    : str = Field(..., description="Ayanamsa system used")
    house_system: str = Field(..., description="House system used")


class PlanetaryPosition(BaseModel):
    """Planetary position details."""
    sign          : str | None   = Field(None, description="Zodiac sign")
    nakshatra     : str | None   = Field(None, description="Nakshatra name")
    nakshatra_pada: int | None   = Field(None, description="Nakshatra pada")
    nakshatra_lord: str | None   = Field(None, description="Nakshatra lord")
    rasi_lord     : str | None   = Field(None, description="Rasi lord")
    sub_lord      : str | None   = Field(None, description="Sub lord")
    sub_sub_lord  : str | None   = Field(None, description="Sub sub lord")
    longitude     : float | None = Field(None, description="Longitude in degrees")


class KeyPositions(BaseModel):
    """Key planetary positions (Sun, Moon, Ascendant)."""
    sun       : PlanetaryPosition = Field(..., description="Sun position")
    moon      : PlanetaryPosition = Field(..., description="Moon position")
    ascendant : PlanetaryPosition = Field(..., description="Ascendant position")
    lagna_lord: str | None        = Field(None, description="Lagna lord (Ascendant lord)")


class PlanetData(BaseModel):
    """Planet data details."""
    object           : str         = Field(..., description="Planet/object name")
    rasi             : str         = Field(..., description="Rasi (zodiac sign)")
    is_retrograde    : bool | None = Field(None, description="Is planet retrograde")
    longitude_dec_deg: float       = Field(..., description="Longitude in decimal degrees")
    sign_lon_dms     : str         = Field(..., description="Sign longitude in DMS format")
    sign_lon_dec_deg : float       = Field(..., description="Sign longitude in decimal degrees")
    lat_dms          : str | None  = Field(None, description="Latitude in DMS format")
    nakshatra        : str | None  = Field(None, description="Nakshatra")
    rasi_lord        : str | None  = Field(None, description="Rasi lord")
    nakshatra_lord   : str | None  = Field(None, description="Nakshatra lord")
    sub_lord         : str | None  = Field(None, description="Sub lord")
    sub_sub_lord     : str | None  = Field(None, description="Sub sub lord")
    house_nr         : int | None  = Field(None, description="House number")


class HouseData(BaseModel):
    """House data details."""
    object           : str        = Field(..., description="House object (Roman numeral)")
    house_nr         : int        = Field(..., description="House number")
    rasi             : str        = Field(..., description="Rasi (zodiac sign)")
    longitude_dec_deg: float      = Field(..., description="Longitude in decimal degrees")
    sign_lon_dms     : str        = Field(..., description="Sign longitude in DMS format")
    sign_lon_dec_deg : float      = Field(..., description="Sign longitude in decimal degrees")
    deg_size         : float      = Field(..., description="Degree size")
    nakshatra        : str | None = Field(None, description="Nakshatra")
    rasi_lord        : str | None = Field(None, description="Rasi lord")
    nakshatra_lord   : str | None = Field(None, description="Nakshatra lord")
    sub_lord         : str | None = Field(None, description="Sub lord")
    sub_sub_lord     : str | None = Field(None, description="Sub sub lord")


class PlanetaryAspect(BaseModel):
    """Planetary aspect details."""
    P1        : str   = Field(..., description="First planet")
    P2        : str   = Field(..., description="Second planet")
    AspectType: str   = Field(..., description="Type of aspect")
    AspectDeg : int   = Field(..., description="Aspect degree")
    AspectOrb : float = Field(..., description="Aspect orb")


class BhuktiDetails(BaseModel):
    """Bhukti (sub-period) details."""
    start: str = Field(..., description="Start date")
    end  : str = Field(..., description="End date")


class DasaDetails(BaseModel):
    """Dasa (major period) details."""
    start  : str                      = Field(..., description="Start date")
    end    : str                      = Field(..., description="End date")
    bhuktis: dict[str, BhuktiDetails] = Field(default_factory=dict, description="Bhukti periods")


class KundaliDetails(BaseModel):
    """Complete kundali details structure."""
    user_name         : str                      = Field(..., description="User name")
    birth_details     : BirthDetails             = Field(..., description="Birth details")
    location          : LocationDetails          = Field(..., description="Location details")
    chart_settings    : ChartSettings            = Field(..., description="Chart calculation settings")
    key_positions     : KeyPositions             = Field(..., description="Key planetary positions")
    planets           : List[PlanetData]         = Field(..., description="List of all planets")
    houses            : List[HouseData]          = Field(..., description="List of all houses")
    planetary_aspects : List[PlanetaryAspect]    = Field(default_factory=list, description="Planetary aspects")
    consolidated_chart: List[dict] | dict | None = Field(None, description="Consolidated chart data")
    vimshottari_dasa  : dict[str, DasaDetails]   = Field(default_factory=dict, description="Vimshottari Dasa periods")



##>=============================================================
##> RAG Query and Retrieval Models
##>=============================================================


# Enums and Literals for RAG Query Output
ZodiacSign = Literal[
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]

PlanetaryFactor = Literal[
    "Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Rahu", "Ketu"
]

LifeArea = Literal["love", "spirituality", "career"]

NakshatraName = Literal[
    "Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira", "Ardra",
    "Punarvasu", "Pushya", "Ashlesha", "Magha", "Purva Phalguni", "Uttara Phalguni",
    "Hasta", "Chitra", "Swati", "Vishakha", "Anuradha", "Jyeshtha",
    "Mula", "Purva Ashadha", "Uttara Ashadha", "Shravana", "Dhanishtha",
    "Shatabhisha", "Purva Bhadrapada", "Uttara Bhadrapada", "Revati"
]


class MetadataFilters(BaseModel):
    """
    Metadata filters for ChromaDB query.
    
    All fields are optional and can be None or empty lists.
    """
    zodiacs          : Optional[List[ZodiacSign]]         = Field(
        default=None,
        description="List of zodiac signs to filter by. Available: Aries, Taurus, Gemini, Cancer, Leo, Virgo, Libra, Scorpio, Sagittarius, Capricorn, Aquarius, Pisces"
    )
    planetary_factors: Optional[List[PlanetaryFactor]]    = Field(
        default=None,
        description="List of planetary factors to filter by. Available: Sun, Moon, Mars, Mercury, Jupiter, Venus, Saturn, Rahu, Ketu"
    )
    life_areas       : Optional[List[LifeArea]]           = Field(
        default=None,
        description="List of life areas to filter by. Available: love, spirituality, career"
    )
    nakshtra         : Optional[List[NakshatraName]]      = Field(
        default=None,
        description="List of nakshatras to filter by. Available: Ashwini, Bharani, Krittika, Rohini, Mrigashira, Ardra, Punarvasu, Pushya, Ashlesha, Magha, Purva Phalguni, Uttara Phalguni, Hasta, Chitra, Swati, Vishakha, Anuradha, Jyeshtha, Mula, Purva Ashadha, Uttara Ashadha, Shravana, Dhanishtha, Shatabhisha, Purva Bhadrapada, Uttara Bhadrapada, Revati"
    )


class RAGQueryOutput(BaseModel):
    """
    Structured output for RAG query generation node.
    
    This model defines the expected output structure when analyzing
    user queries and kundali details to determine if RAG retrieval is needed.
    """
    needs_rag       : bool             = Field(
        ...,
        description="Whether RAG (retrieval) is needed to answer the question. Set to False if the question can be answered with general astrological knowledge or if required information is already present in previous context."
    )
    metadata_filters: Optional[MetadataFilters] = Field(
        default=None,
        description="Metadata filters for ChromaDB query. Only include filters that are relevant to the user's question and kundali details.",examples=[MetadataFilters(zodiacs=["Aries", "Taurus"], planetary_factors=["Sun", "Moon"], life_areas=["love", "career"], nakshtra=["Ashwini", "Bharani"])]
    )
    rag_query       : Optional[str]    = Field(
        default=None,
        description="Optimized search query string for semantic search. Should be concise and focused on the specific astrological information needed. Only provide if needs_rag is True."
    )
    reasoning       : Optional[str]    = Field(
        default=None,
        description="Brief reasoning for why RAG is needed or not needed, and what information is being sought."
    )