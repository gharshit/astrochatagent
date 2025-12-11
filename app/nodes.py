"""
LangGraph nodes for chat flow with RAG.
"""

from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from app.state import GraphState
from app.models import RAGQueryOutput
from app.llmclient import get_structured_llm, get_chat_llm
from helper.utils.logger import setup_logger

logger = setup_logger(name="app.nodes", level=20)  # INFO level


async def context_rag_query_node(state: GraphState) -> GraphState:
    """
    Context node that generates RAG query and metadata filters.
    
    Uses LLM with structured output to analyze user query + kundali details to determine:
    1. If RAG is needed (needs_rag: bool) - checks if information is already in previous context
    2. Metadata filters (zodiacs, planetary_factors, life_areas, nakshtra)
    3. Query string for embedding search
    
    This node checks previous context from checkpointer to avoid unnecessary retrieval
    if the required information is already available.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated graph state with rag_query, rag_context_keys, needs_rag
    """
    logger.info("Processing context/RAG query node...")
    
    user_query = state["messages"][-1].content if state["messages"] else ""
    kundali_details = state.get("kundali_details")
    user_profile = state.get("user_profile")
    
    if not kundali_details or not user_profile:
        logger.warning("Missing kundali_details or user_profile, skipping RAG")
        state["needs_rag"] = False
        state["rag_query"] = None
        state["rag_context_keys"] = []
        state["metadata_filters"] = None
        return state
    
    # Extract key astrological info
    sun_sign       = kundali_details.key_positions.sun.sign or "Unknown"
    moon_sign      = kundali_details.key_positions.moon.sign or "Unknown"
    ascendant_sign = kundali_details.key_positions.ascendant.sign or "Unknown"
    lagna_lord     = kundali_details.key_positions.lagna_lord or "Unknown"
    sun_nakshatra  = kundali_details.key_positions.sun.nakshatra or None
    moon_nakshatra = kundali_details.key_positions.moon.nakshatra or None
    
    # Get planetary positions
    planets_info = []
    for planet in kundali_details.planets:
        planets_info.append(f"{planet.object}: {planet.rasi}")
    
    # Check previous context from checkpointer (previous RAG results)
    previous_rag_results = state.get("rag_results", [])
    previous_context_summary = ""
    if previous_rag_results:
        previous_context_summary = "\n\nPrevious Context Available:\n"
        for i, result in enumerate(previous_rag_results[:3], 1):  # Show first 3 results
            content_preview = result.get("content", "")[:200]  # First 200 chars
            previous_context_summary += f"{i}. {content_preview}...\n"
        previous_context_summary += "\nIMPORTANT: Check if the user's current question can be answered using the previous context above. If yes, set needs_rag to False."
    
    # Get previous context keys
    previous_context_keys = state.get("rag_context_keys", [])
    previous_keys_summary = ""
    if previous_context_keys:
        previous_keys_summary = f"\nPrevious Context Keys Used: {', '.join(previous_context_keys)}"
    
    # LLM prompt with structured output
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert in Vedic astrology. Analyze the user's question and their kundali details to determine:

1. If RAG (retrieval) is needed to answer the question
   - Check if the question can be answered with general astrological knowledge
   - Check if required information is already present in previous context (if provided)
   - Only set needs_rag=True if specific astrological information needs to be retrieved

2. What metadata filters to use (only if needs_rag=True):
   - zodiacs: CRITICAL - ONLY use the native's Sun Sign, Moon Sign, or Ascendant from the kundali. DO NOT use zodiacs from planetary positions. Only include zodiacs that are directly relevant to answering the question (e.g., if question is about personality, use sun sign; if about emotions, use moon sign; if about general life, use ascendant).
   - planetary_factors: Filter by planets mentioned or relevant to the question, fetch proper planets info which can be used to reason on the question (e.g., Sun, Moon, Mars, Mercury, Jupiter, Venus, Saturn, Rahu, Ketu)
   - life_areas: Filter by life area (love, spirituality, career) if question is about that area
   - nakshtra: Filter by nakshatra if question is specifically about nakshatra (use Sun Nakshatra or Moon Nakshatra from kundali)

3. A search query optimized for semantic search (only if needs_rag=True)
   - Should be concise and focused on the specific astrological information needed
   - Include relevant astrological terms and concepts

Available Zodiac Signs: Aries, Taurus, Gemini, Cancer, Leo, Virgo, Libra, Scorpio, Sagittarius, Capricorn, Aquarius, Pisces
Available Planetary Factors: Sun, Moon, Mars, Mercury, Jupiter, Venus, Saturn, Rahu, Ketu
Available Life Areas: love, spirituality, career
Available Nakshatras: Ashwini, Bharani, Krittika, Rohini, Mrigashira, Ardra, Punarvasu, Pushya, Ashlesha, Magha, Purva Phalguni, Uttara Phalguni, Hasta, Chitra, Swati, Vishakha, Anuradha, Jyeshtha, Mula, Purva Ashadha, Uttara Ashadha, Shravana, Dhanishtha, Shatabhisha, Purva Bhadrapada, Uttara Bhadrapada, Revati

User's Kundali:
- Sun Sign: {sun_sign}
- Moon Sign: {moon_sign}
- Ascendant: {ascendant_sign}
- Lagna Lord: {lagna_lord}
- Sun Nakshatra: {sun_nakshatra}
- Moon Nakshatra: {moon_nakshatra}
- Planets: {planets_info}

CRITICAL RULES FOR ZODIAC FILTERS:
- For zodiacs filter, ONLY use the native's Sun Sign ({sun_sign}), Moon Sign ({moon_sign}), or Ascendant ({ascendant_sign})
- DO NOT use zodiacs from planetary positions (e.g., Jupiter in Sagittarius, Saturn in Pisces, etc.)
- Only include zodiacs that are directly relevant to the question:
  * Use Sun Sign for personality, ego, identity questions
  * Use Moon Sign for emotions, feelings, mental state questions
  * Use Ascendant for general life, appearance, first impressions questions
- Maximum 1-3 zodiacs should be included, and they must be from the native's Sun/Moon/Ascendant only

{previous_context_summary}{previous_keys_summary}

Guidelines for Handling Inappropriate Queries:
- If the user's question contains NSFW content, dangerous requests, harmful instructions, or topics unrelated to astrology, set needs_rag=False and rag_query=None
- Do not process queries that request illegal activities, self-harm, violence, or explicit adult content
- For non-astrology related queries, politely decline and redirect to astrology topics
- Maintain professional boundaries and ethical standards at all times
- If a query is ambiguous but potentially harmful, err on the side of caution and set needs_rag=False

Output Variables:
    "needs_rag"       : bool,                   # True only if the information is not present in context or previous RAG results and need extra/additional information from ChromaDB.
    "metadata_filters": List| None,             # Metadata filters for ChromaDB query. Only include filters that are relevant to the user's question and kundali details and if needs_rag is True
    "rag_query"       : str | None,             # Optimized search query string for semantic search. Should be concise and focused on the specific astrological information needed. Only provide if needs_rag is True.
    "reasoning"       : str | None              # Brief reasoning for why RAG is needed or not needed, and what information is being sought. Only provide if needs_rag is True.
"""),
        
    ("human", "User Question: {user_query}")
    ])
    
    # Use structured LLM with lower temperature for deterministic output
    llm = get_structured_llm()
    structured_llm = llm.with_structured_output(RAGQueryOutput)
    chain = prompt | structured_llm
    
    try:
        result = await chain.ainvoke({
            "user_query"              : user_query,
            "sun_sign"                : sun_sign,
            "moon_sign"               : moon_sign,
            "ascendant_sign"          : ascendant_sign,
            "lagna_lord"              : lagna_lord,
            "sun_nakshatra"           : sun_nakshatra or "Unknown",
            "moon_nakshatra"          : moon_nakshatra or "Unknown",
            "planets_info"            : ", ".join(planets_info),
            "previous_context_summary": previous_context_summary,
            "previous_keys_summary"   : previous_keys_summary
        })
        
        # Extract values from structured output
        state["needs_rag"] = result.needs_rag
        state["rag_query"] = result.rag_query
        
        # Extract context keys from metadata filters
        context_keys = []
        metadata_filters_dict = None
        
        if result.metadata_filters:
            metadata_filters_dict = {}
            
            # Validate and filter zodiacs to only include native's sun, moon, or ascendant
            native_zodiacs = {sun_sign, moon_sign, ascendant_sign}
            if result.metadata_filters.zodiacs:
                # Only keep zodiacs that match native's sun, moon, or ascendant
                valid_zodiacs = [z for z in result.metadata_filters.zodiacs if z in native_zodiacs]
                if valid_zodiacs:
                    context_keys.extend([f"zodiacs:{z}" for z in valid_zodiacs])
                    metadata_filters_dict["zodiacs"] = valid_zodiacs
                else:
                    logger.warning(f"Filtered out invalid zodiacs: {result.metadata_filters.zodiacs}. Only using native zodiacs: {native_zodiacs}")
            
            if result.metadata_filters.planetary_factors:
                context_keys.extend([f"planetary_factors:{p}" for p in result.metadata_filters.planetary_factors])
                metadata_filters_dict["planetary_factors"] = result.metadata_filters.planetary_factors
            
            if result.metadata_filters.life_areas:
                context_keys.extend([f"life_areas:{a}" for a in result.metadata_filters.life_areas])
                metadata_filters_dict["life_areas"] = result.metadata_filters.life_areas
            
            if result.metadata_filters.nakshtra:
                context_keys.extend([f"nakshtra:{n}" for n in result.metadata_filters.nakshtra])
                metadata_filters_dict["nakshtra"] = result.metadata_filters.nakshtra
        
        state["rag_context_keys"] = context_keys
        state["metadata_filters"] = metadata_filters_dict
        
        logger.info(f"RAG needed: {state['needs_rag']}, Context keys: {context_keys}")
        if result.reasoning:
            logger.info(f"Reasoning: {result.reasoning}")
        
    except Exception as e:
        logger.error(f"Error in context node: {str(e)}", exc_info=True)
        state["needs_rag"] = False
        state["rag_query"] = None
        state["rag_context_keys"] = []
        state["metadata_filters"] = None
    
    return state


async def retrieval_node(state: GraphState, config: RunnableConfig | None = None) -> GraphState:
    """
    Retrieval node that fetches relevant documents from ChromaDB.
    
    Uses rag_query and metadata_filters to query ChromaDB.
    Updates rag_results and rag_context_keys (replaced, not appended).
    
    Args:
        state: Current graph state
        config: LangGraph RunnableConfig containing query_function
        
    Returns:
        Updated graph state with rag_results and updated rag_context_keys
    """
    logger.info("Processing retrieval node...")
    
    if not state.get("needs_rag") or not state.get("rag_query"):
        logger.info("RAG not needed, skipping retrieval")
        state["rag_results"] = []
        return state
    
    # Get query_function from config
    query_function = None
    if config and "configurable" in config:
        query_function = config["configurable"].get("query_function")
    
    if not query_function:
        logger.error("Query function not available in config")
        state["rag_results"] = []
        return state
    
    metadata_filters = state.get("metadata_filters", {})
    rag_query = state["rag_query"]
    
    logger.info(f"Metadata filters (dict): {metadata_filters}")
    
    # Build where clause for metadata filtering
    # IMPORTANT: Each document in ChromaDB has only ONE metadata field populated
    # (either zodiacs, planetary_factors, life_areas, or nakshtra)
    # Therefore, we use $or to match documents that have ANY of the specified filters
    # This allows us to retrieve documents matching any relevant category
    where_clause = None
    conditions = []
    
    if metadata_filters:
        if metadata_filters.get("zodiacs"):
            conditions.append({"zodiacs": {"$in": metadata_filters["zodiacs"]}})
        if metadata_filters.get("planetary_factors"):
            conditions.append({"planetary_factors": {"$in": metadata_filters["planetary_factors"]}})
        if metadata_filters.get("life_areas"):
            conditions.append({"life_areas": {"$in": metadata_filters["life_areas"]}})
        if metadata_filters.get("nakshtra"):
            conditions.append({"nakshtra": {"$in": metadata_filters["nakshtra"]}})
        
        # Use $or since documents only have one metadata field each
        # This matches documents that have ANY of the specified filters
        if len(conditions) == 1:
            where_clause = conditions[0]
        elif len(conditions) > 1:
            where_clause = {"$or": conditions}
    
    try:
        # Query ChromaDB
        results = query_function(
            query_text=rag_query,
            n_results=3,
            where=where_clause
        )
        
        logger.info(f"\n Where clause: {where_clause}")
        logger.info(f"Rag query: {rag_query}")
        logger.info(f"Results preview: {results} \n")
        
        # Extract documents and metadata
        rag_results = []
        context_keys = []
        
        if results and "documents" in results and results["documents"]:
            documents = results["documents"][0] if results["documents"] else []
            metadatas = results.get("metadatas", [[]])
            metadata_list = metadatas[0] if metadatas and len(metadatas) > 0 else []
            
            for i, doc in enumerate(documents):
                # Safely extract metadata
                metadata = metadata_list[i] if i < len(metadata_list) else {}
                
                rag_results.append({
                    "content": doc,
                    "metadata": metadata
                })
                
                # Extract context keys from metadata
                if metadata and isinstance(metadata, dict):
                    if metadata.get("zodiacs"):
                        context_keys.append(f"zodiacs:{metadata['zodiacs']}")
                    if metadata.get("planetary_factors"):
                        context_keys.append(f"planetary_factors:{metadata['planetary_factors']}")
                    if metadata.get("life_areas"):
                        context_keys.append(f"life_areas:{metadata['life_areas']}")
                    if metadata.get("nakshtra"):
                        context_keys.append(f"nakshtra:{metadata['nakshtra']}")
        
        # Update state (replace, not append)
        state["rag_results"] = rag_results
        state["rag_context_keys"] = list(set(context_keys))  # Unique keys
        
        logger.info(f"Retrieved {len(rag_results)} documents from ChromaDB")
        logger.info(f"Context keys: {state['rag_context_keys']}")
        
    except Exception as e:
        logger.error(f"Error in retrieval node: {str(e)}", exc_info=True)
        state["rag_results"] = []
    
    return state


async def chat_node(state: GraphState) -> GraphState:
    """
    Chat node that generates final personalized response.
    
    Uses all available information:
    - User query (messages)
    - Kundali details
    - RAG results (if any)
    - Preferred language
    
    Generates response in the user's preferred language.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated graph state with AI response message
    """
    logger.info("Processing chat node...")
    
    user_query = state["messages"][-1].content if state["messages"] else ""
    kundali_details = state.get("kundali_details")
    user_profile = state.get("user_profile")
    rag_results = state.get("rag_results", [])
    rag_context_keys = state.get("rag_context_keys", [])
    
    preferred_language = user_profile.preffered_language if user_profile else "en"
    language_name = "Hindi" if preferred_language == "hi" else "English"
    
    # Extract kundali summary
    kundali_summary = ""
    if kundali_details:
        sun_sign = kundali_details.key_positions.sun.sign or "Unknown"
        moon_sign = kundali_details.key_positions.moon.sign or "Unknown"
        ascendant = kundali_details.key_positions.ascendant.sign or "Unknown"
        kundali_summary = f"Sun Sign: {sun_sign}, Moon Sign: {moon_sign}, Ascendant: {ascendant}"
    
    # Prepare RAG context
    rag_context = ""
    if rag_results:
        rag_context = "\n\nRelevant Astrological Information:\n"
        for i, result in enumerate(rag_results, 1):
            rag_context += f"{i}. {result['content']}\n"
    
    # Context keys summary
    context_summary = ""
    if rag_context_keys:
        zodiacs = [k.split(":")[1] for k in rag_context_keys if k.startswith("zodiacs:")]
        planets = [k.split(":")[1] for k in rag_context_keys if k.startswith("planetary_factors:")]
        areas = [k.split(":")[1] for k in rag_context_keys if k.startswith("life_areas:")]
        
        context_summary = "Context used: "
        if zodiacs:
            context_summary += f"Zodiacs: {', '.join(zodiacs)}. "
        if planets:
            context_summary += f"Planets: {', '.join(planets)}. "
        if areas:
            context_summary += f"Life Areas: {', '.join(areas)}. "
    
    # LLM prompt for final response
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are an expert Vedic astrologer with decades of experience in Jyotish Shastra (Vedic astrology). You embody the wisdom of ancient Indian astrological traditions, combining deep knowledge of planetary influences, nakshatras, dasas, and kundali analysis with compassionate guidance.

Your Persona:
- You are a learned scholar who has studied classical texts like Brihat Parashara Hora Shastra, Jataka Parijata, and Phaladeepika
- You approach each consultation with reverence for the cosmic influences and respect for the individual's unique astrological chart
- You speak with the authority of traditional knowledge while maintaining warmth and empathy
- You understand that astrology is a tool for self-awareness and guidance, not deterministic fate
- You provide insights that empower users to make informed decisions while respecting free will
- You maintain a balanced perspective, acknowledging both strengths and challenges indicated in the chart
- You use traditional Vedic terminology appropriately while making concepts accessible

Response Guidelines:
- Keep responses concise: Aim for 20-30 words maximum. Be direct and impactful
- Maintain an optimistic and encouraging tone: Focus on positive aspects, growth opportunities, and constructive guidance
- Use proper, respectful language: Address the user with dignity and maintain professional boundaries
- Reference specific astrological placements: Mention relevant planets, signs, houses, or nakshatras when applicable
- Provide actionable insights: Offer practical guidance that users can apply to their lives
- Stay within astrological scope: Focus on astrological interpretations and avoid medical, legal, or financial advice

User's Kundali Details:
{kundali_summary}
{rag_context}

IMPORTANT: You MUST respond strictly in {language_name} ({preferred_language}).

Guidelines for Handling Inappropriate Queries:
- If the user's question contains NSFW content, dangerous requests, harmful instructions, or topics unrelated to astrology, respond politely but firmly
- Decline requests for illegal activities, self-harm guidance, violence, or explicit adult content with a professional boundary statement
- For non-astrology related queries, politely redirect: "I specialize in Vedic astrology guidance. How can I help you understand your kundali better?"
- Maintain ethical standards: Do not provide guidance that could cause harm or encourage dangerous behavior
- If a query is inappropriate, respond formally: "I'm here to provide astrological guidance based on your kundali. I cannot assist with that request, but I'm happy to help with astrology-related questions."

{context_summary}"""),
        ("human", "{user_query}")
    ])
    
    llm = get_chat_llm()
    chain = prompt | llm
    
    try:
        response = await chain.ainvoke({"user_query": user_query})
        
        # Add AI response to messages
        state["messages"].append(AIMessage(content=response.content))
        
        logger.info(f"Generated response in {language_name}")
        
    except Exception as e:
        logger.error(f"Error in chat node: {str(e)}", exc_info=True)
        # Fallback response
        fallback_msg = "मुझे क्षमा करें, मैं आपकी क्वेरी को संसाधित करने में असमर्थ था।" if preferred_language == "hi" else "I apologize, I was unable to process your query."
        state["messages"].append(AIMessage(content=fallback_msg))
    
    return state
