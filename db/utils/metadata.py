"""
Metadata creation utilities for different content types.
"""

from typing import Dict, Any, Optional
from .logger import logger


def create_metadata(filename: str, main_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Create appropriate metadata based on file type.

    Args:
        filename (str): Name of the file (without extension)
        main_key (str, optional): Main key from JSON structure

    Returns:
        Dict[str, Any]: Metadata dictionary with appropriate keys:
            - zodiacs: For zodiac-related content
            - planetary_factors: For planetary impact content
            - nakshtra: For nakshtra content
            - life_areas: For life guidance content (love, spirituality, career)
            - content_type: Type of content
    """
    metadata = {}

    if filename == "zodiac_traits":
        metadata["zodiacs"] = main_key if main_key else "general"
        metadata["content_type"] = "zodiac_traits"
        logger.debug(f"Created metadata for zodiac_traits with zodiac: {main_key}")
    elif filename == "planetary_impact":
        metadata["planetary_factors"] = main_key if main_key else "general"
        metadata["content_type"] = "planetary_impact"
        logger.debug(f"Created metadata for planetary_impact with planet: {main_key}")
    elif filename == "nakshtras":
        metadata["nakshtra"] = main_key if main_key else "general"
        metadata["content_type"] = "nakshtras"
        logger.debug(f"Created metadata for nakshtras with nakshtra: {main_key}")
    elif filename in ["love_guidance", "spiritual_guidance", "carrer_guidance"]:
        life_area_map = {
            "love_guidance"     : "love",
            "spiritual_guidance": "spirituality",
            "carrer_guidance"   : "career"
        }
        metadata["life_areas"] = life_area_map.get(filename, "general")
        metadata["content_type"] = "life_guidance"
        logger.debug(f"Created metadata for life_guidance with area: {metadata['life_areas']}")
    else:
        metadata["content_type"] = "general"
        logger.debug(f"Created generic metadata for {filename}")

    return metadata

