"""
Chat router for handling user chat requests with LangGraph checkpoint management.

This router uses LangGraph's checkpoint system to automatically manage
conversation state and memory per session.
"""

from fastapi import APIRouter, Request, HTTPException, status
from datetime import datetime
from app.models import ChatRequest, ChatResponse, KundaliDetails
from app.utils import fetch_kundali_details
from app.state import GraphState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from helper.utils.logger import setup_logger

# Setup logger for chat router
logger = setup_logger(name="app.router.chat", level=20)  # INFO level


chat_router = APIRouter(
    prefix="/v1/chat",
    tags=["chat"]
)


@chat_router.post(
    "/",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    description="Chat with the user using LangGraph with automatic state management",
    responses={
        200: {
            "description": "Chat response generated successfully",
            "content": {
                "application/json": {
                    "example": {
                        "response": "Hello, World!",
                        "user_profile": {...},
                        "kundali_details": {...}
                    }
                }
            }
        },
        400: {
            "description": "Invalid request data (user profile validation failed)"
        },
        503: {
            "description": "LangGraph service not available"
        },
        500: {
            "description": "Internal server error during chat processing"
        }
    }
)
async def chat(chat_request: ChatRequest, request: Request) -> dict:
    """
    Chat with the user using LangGraph with automatic checkpoint management.
    
    LangGraph's checkpoint system automatically:
    - Persists state for each thread_id (session_id)
    - Restores state when the same thread_id is used
    - Manages conversation history across requests
    
    This endpoint:
    1. Checks if compiled graph is available
    2. Checks if state exists for this session_id (thread_id)
    3. If new session: Fetches kundali details and initializes state
    4. Invokes graph with message, which automatically persists state
    5. Returns chat response with user profile and kundali details
    
    Args:
        chat_request: Chat request containing:
            - session_id: Unique session identifier (used as thread_id)
            - message: User's message/question
            - user_profile: User profile with birth details
        request: FastAPI request object for accessing app state
        
    Returns:
        dict: Response containing:
            - response: Chat response message
            - user_profile: User profile data
            - kundali_details: Complete kundali details
            
    Raises:
        HTTPException:
            - 400: If user profile validation fails
            - 503: If LangGraph service is not available
            - 500: If there's an error processing the chat request
    """
    logger.info("=" * 60)
    logger.info("Received chat request")
    logger.info(f"Session ID (thread_id): {chat_request.session_id}")
    logger.info(f"User: {chat_request.user_profile.name}")
    logger.info(f"Message: {chat_request.message}")
    logger.info("=" * 60)
    
    try:
        # Check if compiled graph is available
        if not hasattr(request.app.state, 'compiled_graph') or request.app.state.compiled_graph is None:
            logger.error("Compiled graph not initialized")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="LangGraph service not available"
            )
        
        compiled_graph = request.app.state.compiled_graph
        checkpoint_memory: MemorySaver = request.app.state.checkpoint_memory
        
        # LangGraph uses thread_id to manage separate conversation states
        thread_id = chat_request.session_id
        logger.info(f"Processing request for thread_id: {thread_id}")
        
        # Try to get existing state from checkpoint
        # LangGraph automatically restores state when we invoke with the same thread_id
        # But we need to check if kundali_details exists before invoking
        kundali_details: KundaliDetails | None = None
        try:
            checkpoint = await checkpoint_memory.aget({"configurable": {"thread_id": thread_id}})
            if checkpoint and checkpoint.get("channel_values"):
                existing_state = checkpoint["channel_values"]
                kundali_details_dict = existing_state.get("kundali_details")
                if kundali_details_dict:
                    logger.info("✓ Found existing state with kundali details")
                    if isinstance(kundali_details_dict, dict):
                        kundali_details = KundaliDetails(**kundali_details_dict)
                    else:
                        kundali_details = kundali_details_dict
        except Exception:
            # No existing checkpoint, this is a new thread
            logger.info("No existing checkpoint found, new thread")
        
        # Fetch kundali details only if not found in existing state
        if kundali_details is None:
            logger.info("Fetching kundali details for new session...")
            logger.debug(f"Birth details - Date: {chat_request.user_profile.birth_date}, "
                        f"Time: {chat_request.user_profile.birth_time}, "
                        f"Place: {chat_request.user_profile.birth_place}")
            
            kundali_details = await fetch_kundali_details(chat_request.user_profile, request)
            logger.info(f"✓ Kundali fetched - Sun: {kundali_details.key_positions.sun.sign or 'N/A'}, "
                       f"Moon: {kundali_details.key_positions.moon.sign or 'N/A'}")
        
        # Prepare initial state with new message
        # LangGraph will automatically:
        # - Merge with existing state if thread exists (preserving conversation history)
        # - Create new state if thread doesn't exist
        # - Persist state after execution
        initial_state: GraphState = {
            "messages"        : [HumanMessage(content=chat_request.message)],
            "user_profile"    : chat_request.user_profile,
            "kundali_details" : kundali_details,
            "session_id"      : thread_id,
            "rag_context_keys": [],
            "rag_query"       : None,
            "rag_results"     : [],
            "needs_rag"       : False,
            "metadata_filters": None
        }
        
        # Get query_function from app state
        query_function = request.app.state.query_function
        
        # Invoke graph with checkpoint configuration
        # LangGraph automatically handles state restoration and persistence
        logger.info("Invoking LangGraph (state will be auto-restored/persisted)...")
        config = {
            "configurable": {
                "thread_id": thread_id,
                "query_function": query_function
            }
        }
        
        final_state = await compiled_graph.ainvoke(initial_state, config=config)
        
        logger.info("✓ Graph execution completed, state automatically persisted")
        
        # Extract response from final state
        response      = final_state.get("messages", [])[-1].content
        context_used  = final_state.get("rag_context_keys", [])
        
        # Extract astrological details from kundali
        sun_sign      = kundali_details.key_positions.sun.sign or "Unknown"
        moon_sign     = kundali_details.key_positions.moon.sign or "Unknown"
        ascendant_sign = kundali_details.key_positions.ascendant.sign or "Unknown"
        
        # Extract and format current dasha information
        dasha_info = "Not available"
        vimshottari_dasa = getattr(kundali_details, 'vimshottari_dasa', None)
        if vimshottari_dasa:
            try:
                # Helper function to parse date in DD-MM-YYYY format
                def parse_dasha_date(date_str: str):
                    """Parse date string in DD-MM-YYYY format."""
                    try:
                        return datetime.strptime(date_str, "%d-%m-%Y").date()
                    except ValueError:
                        # Try YYYY-MM-DD format as fallback
                        try:
                            return datetime.strptime(date_str, "%Y-%m-%d").date()
                        except ValueError:
                            raise ValueError(f"Unable to parse date: {date_str}")
                
                # Get current date
                current_date = datetime.now().date()
                
                # Find the current dasa (the one that contains today's date)
                current_dasa_name = None
                current_dasa_data = None
                
                for dasa_name, dasa_data in vimshottari_dasa.items():
                    try:
                        dasa_start = parse_dasha_date(dasa_data.start)
                        dasa_end = parse_dasha_date(dasa_data.end)
                        
                        # Check if current date falls within this dasa period
                        if dasa_start <= current_date <= dasa_end:
                            current_dasa_name = dasa_name
                            current_dasa_data = dasa_data
                            break
                    except (ValueError, AttributeError) as e:
                        logger.debug(f"Error parsing dasa dates for {dasa_name}: {e}")
                        continue
                
                # If current dasa found, format it with current bhukti
                if current_dasa_name and current_dasa_data:
                    dasa_str = f"{current_dasa_name} ({current_dasa_data.start} to {current_dasa_data.end})"
                    
                    # Find the current bhukti within the current dasa
                    if current_dasa_data.bhuktis:
                        current_bhukti_name = None
                        current_bhukti_data = None
                        
                        for bhukti_name, bhukti_data in current_dasa_data.bhuktis.items():
                            try:
                                bhukti_start = parse_dasha_date(bhukti_data.start)
                                bhukti_end = parse_dasha_date(bhukti_data.end)
                                
                                # Check if current date falls within this bhukti period
                                if bhukti_start <= current_date <= bhukti_end:
                                    current_bhukti_name = bhukti_name
                                    current_bhukti_data = bhukti_data
                                    break
                            except (ValueError, AttributeError) as e:
                                logger.debug(f"Error parsing bhukti dates for {bhukti_name}: {e}")
                                continue
                        
                        # Add current bhukti info if found
                        if current_bhukti_name and current_bhukti_data:
                            dasa_str += f" - Current Bhukti: {current_bhukti_name} ({current_bhukti_data.start} to {current_bhukti_data.end})"
                    
                    dasha_info = dasa_str
                    
            except Exception as e:
                logger.warning(f"Error extracting dasha info: {str(e)}", exc_info=True)
                dasha_info = "Not available"
        
        return ChatResponse(
            response       = response,
            context_used   = context_used,
            sun_sign       = sun_sign,
            moon_sign      = moon_sign,
            ascendant_sign = ascendant_sign,
            dasha_info     = dasha_info
        )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )
