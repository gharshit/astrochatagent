"""
LangGraph graph builder.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from app.state import GraphState
from app.nodes import chat_node, initialize_session_node
from helper.utils.logger import setup_logger

logger = setup_logger(name="app.builder", level=20)  # INFO level


def build_graph() -> StateGraph:
    """
    Build the LangGraph state graph for chat flow.
    
    Returns:
        StateGraph: Uncompiled graph
    """
    logger.info("Building LangGraph...")
    
    # Create the graph
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("initialize_session", initialize_session_node)
    workflow.add_node("chat", chat_node)
    
    # Set entry point
    workflow.set_entry_point("initialize_session")
    
    # Add edges
    workflow.add_edge("initialize_session", "chat")
    workflow.add_edge("chat", END)
    
    logger.info("✓ Graph built successfully")
    return workflow


def compile_graph(checkpoint_memory: MemorySaver) -> StateGraph:
    """
    Compile the LangGraph with checkpoint memory.
    
    When compiled with checkpoint_memory, LangGraph automatically:
    - Persists state for each thread_id (session_id)
    - Restores state when the same thread_id is used
    - Manages conversation history across requests
    
    Args:
        checkpoint_memory: MemorySaver instance for state persistence
        
    Returns:
        Compiled StateGraph ready for execution
    """
    logger.info("Compiling LangGraph with checkpoint memory...")
    
    workflow = build_graph()
    
    # Compile with checkpoint memory
    # The checkpoint_memory will automatically handle state persistence
    # Each thread_id (session_id) maintains its own state
    compiled_graph = workflow.compile(checkpointer=checkpoint_memory)
    
    logger.info("✓ Graph compiled successfully with checkpoint memory")
    return compiled_graph

