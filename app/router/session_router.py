"""
Session router for managing user sessions and retrieving stored kundali details.

This router provides endpoints to retrieve kundali details for existing sessions
stored in checkpoint memory.
"""

from fastapi import APIRouter, Request, HTTPException, status, Path
from app.models import KundaliDetails
from helper.utils.logger import setup_logger
from typing import Dict, Any

# Setup logger for session router
logger = setup_logger(name="app.router.session", level=20)  # INFO level


session_router = APIRouter(
    prefix="/v1/session",
    tags=["session"]
)


@session_router.get(
    "/{session_id}/kundali",
    response_model=KundaliDetails,
    status_code=status.HTTP_200_OK,
    description="Retrieve kundali details for an existing session",
    responses={
        200: {
            "description": "Kundali details retrieved successfully",
            "model": KundaliDetails
        },
        404: {
            "description": "Session not found"
        },
        503: {
            "description": "Session management service not available"
        }
    }
)
async def get_session_kundali(
    request: Request,
    session_id: str = Path(..., description="Session ID to retrieve kundali details for")
) -> KundaliDetails:
    """
    Retrieve kundali details for an existing session.
    
    This endpoint looks up a session by ID in the checkpoint memory
    and returns the stored kundali details if the session exists.
    
    Args:
        session_id: Unique session identifier (path parameter)
        request: FastAPI request object for accessing app state
        
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
            - 404: If session ID is not found in checkpoint memory
            - 503: If session management service is not available
    """
    logger.info("=" * 60)
    logger.info(f"Received request to retrieve kundali for session: {session_id}")
    logger.info("=" * 60)
    
    try:
        # Check if checkpoint memory is available
        if not hasattr(request.app.state, 'checkpoint_memory'):
            logger.error("Checkpoint memory not initialized")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Session management service not available"
            )
        
        checkpoint_memory = request.app.state.checkpoint_memory
        
        # Check if session exists using LangGraph's checkpoint system
        logger.info(f"Checking if thread exists: {session_id}")
        existing_threads = await checkpoint_memory.alist()
        thread_exists = any(
            str(thread) == session_id or (hasattr(thread, 'thread_id') and thread.thread_id == session_id)
            for thread in existing_threads
        )
        
        if not thread_exists:
            logger.warning(f"Session not found: {session_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session ID '{session_id}' not found"
            )
        
        # Retrieve session data from LangGraph checkpoint
        logger.info("Session found. Retrieving kundali details from checkpoint...")
        checkpoint = await checkpoint_memory.aget({"configurable": {"thread_id": session_id}})
        
        if not checkpoint or not checkpoint.get("channel_values"):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"State not found for session ID '{session_id}'"
            )
        
        state = checkpoint["channel_values"]
        kundali_details_dict = state.get("kundali_details")
        
        if not kundali_details_dict:
            logger.error(f"Kundali details not found in session: {session_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Kundali details not found for session ID '{session_id}'"
            )
        
        # Convert dict to Pydantic model
        try:
            kundali_details = KundaliDetails(**kundali_details_dict)
        except Exception as e:
            logger.error(f"Error parsing kundali details: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error parsing stored kundali details: {str(e)}"
            )
        
        logger.info("✓ Kundali details retrieved successfully")
        logger.info(f"User: {kundali_details.user_name}")
        logger.info(f"Sun Sign: {kundali_details.key_positions.sun.sign or 'N/A'}")
        logger.info(f"Moon Sign: {kundali_details.key_positions.moon.sign or 'N/A'}")
        logger.info("=" * 60)
        
        return kundali_details
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error retrieving session kundali: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@session_router.get(
    "/{session_id}",
    status_code=status.HTTP_200_OK,
    description="Get session information",
    responses={
        200: {
            "description": "Session information retrieved successfully"
        },
        404: {
            "description": "Session not found"
        },
        503: {
            "description": "Session management service not available"
        }
    }
)
async def get_session_info(
    request: Request,
    session_id: str = Path(..., description="Session ID to retrieve information for")
) -> dict:
    """
    Get session information including user profile and kundali summary.
    
    Args:
        session_id: Unique session identifier (path parameter)
        request: FastAPI request object for accessing app state
        
    Returns:
        dict: Session information containing:
            - session_id: Session identifier
            - user_profile: User profile data
            - kundali_summary: Summary of kundali details (key positions only)
            
    Raises:
        HTTPException:
            - 404: If session ID is not found
            - 503: If session management service is not available
    """
    logger.info(f"Received request for session info: {session_id}")
    
    try:
        # Check if checkpoint memory is available
        if not hasattr(request.app.state, 'checkpoint_memory'):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Session management service not available"
            )
        
        checkpoint_memory = request.app.state.checkpoint_memory
        
        # Check if session exists using LangGraph's checkpoint system
        existing_threads = await checkpoint_memory.alist()
        thread_exists = any(
            str(thread) == session_id or (hasattr(thread, 'thread_id') and thread.thread_id == session_id)
            for thread in existing_threads
        )
        
        if not thread_exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session ID '{session_id}' not found"
            )
        
        # Retrieve state from checkpoint
        checkpoint = await checkpoint_memory.aget({"configurable": {"thread_id": session_id}})
        if not checkpoint or not checkpoint.get("channel_values"):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"State not found for session ID '{session_id}'"
            )
        
        state = checkpoint["channel_values"]
        user_profile = state.get("user_profile", {})
        kundali_details_dict = state.get("kundali_details", {})
        
        # Extract key positions summary
        kundali_summary = {}
        if kundali_details_dict and "key_positions" in kundali_details_dict:
            key_pos = kundali_details_dict["key_positions"]
            kundali_summary = {
                "sun_sign"      : key_pos.get("sun", {}).get("sign"),
                "moon_sign"     : key_pos.get("moon", {}).get("sign"),
                "ascendant_sign": key_pos.get("ascendant", {}).get("sign"),
                "lagna_lord"    : key_pos.get("lagna_lord")
            }
        
        logger.info("✓ Session info retrieved successfully")
        
        return {
            "session_id"     : session_id,
            "user_profile"   : user_profile,
            "kundali_summary": kundali_summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving session info: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

