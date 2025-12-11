"""
LangGraph nodes for chat flow.
"""

from app.state import GraphState
from helper.utils.logger import setup_logger

logger = setup_logger(name="app.nodes", level=20)  # INFO level


async def chat_node(state: GraphState) -> GraphState:
    """
    Chat node that processes user messages.
    
    This node receives the current state and processes the chat message.
    The state is automatically persisted by LangGraph's checkpoint system.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated graph state
    """
    logger.info(f"Processing chat node for session: {state.get('session_id', 'unknown')}")
    logger.info(f"Messages count: {len(state.get('messages', []))}")
    
    # TODO: Implement actual chat logic here
    # For now, just pass through the state
    # In the future, this will:
    # 1. Use kundali_details to generate personalized responses
    # 2. Query ChromaDB for relevant astrological information
    # 3. Generate response using LLM
    
    return state


async def retrieval_context_node(state: GraphState) -> GraphState:
    """
    Retrieval context node that retrieves the context used to generate the response.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated graph state with user_profile and kundali_details
    """
    logger.info(f"Initializing session node for session: {state.get('session_id', 'unknown')}")
    
    # The user_profile and kundali_details should already be set in state
    # This node can perform any additional initialization logic
    
    return state

