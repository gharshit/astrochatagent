"""
LangGraph graph builder with RAG flow.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from app.state import GraphState
from app.nodes import context_rag_query_node, retrieval_node, chat_node
from helper.utils.logger import setup_logger

logger = setup_logger(name="app.builder", level=20)  # INFO level


def should_retrieve(state: GraphState) -> str:
    """
    Conditional edge function to determine if retrieval is needed.
    
    Returns:
        "retrieve" if needs_rag is True, "chat" otherwise
    """
    if state.get("needs_rag", False):
        return "retrieve"
    return "chat"


def build_graph() -> StateGraph:
    """
    Build the LangGraph state graph for chat flow with RAG.
    
    Flow:
    1. context_rag_query_node: Generates RAG query and metadata filters
    2. Conditional: If needs_rag -> retrieval_node, else -> chat_node
    3. retrieval_node: Fetches documents from ChromaDB
    4. chat_node: Generates final response
    
    Returns:
        StateGraph: Uncompiled graph
    """
    logger.info("Building LangGraph with RAG flow...")
    
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("context_query", context_rag_query_node)
    workflow.add_node("retrieve", retrieval_node)
    workflow.add_node("chat", chat_node)
    
    # Set entry point
    workflow.set_entry_point("context_query")
    
    # Add conditional edge from context_query
    workflow.add_conditional_edges(
        "context_query",
        should_retrieve,
        {
            "retrieve": "retrieve",
            "chat": "chat"
        }
    )
    
    # After retrieval, go to chat
    workflow.add_edge("retrieve", "chat")
    
    # Chat is the end
    workflow.add_edge("chat", END)
    
    logger.info("✓ Graph built successfully with RAG flow")
    return workflow


def compile_graph(checkpoint_memory: MemorySaver) -> StateGraph:
    """
    Compile the LangGraph with checkpoint memory.
    
    Args:
        checkpoint_memory: MemorySaver instance for state persistence
        
    Returns:
        Compiled StateGraph ready for execution
    """
    logger.info("Compiling LangGraph with checkpoint memory...")
    
    workflow = build_graph()
    compiled_graph = workflow.compile(checkpointer=checkpoint_memory)
    
    logger.info("✓ Graph compiled successfully with checkpoint memory")
    return compiled_graph
