"""
LangGraph nodes for chat flow with RAG.
"""

from typing import List, Dict, Any
from datetime import datetime
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
    
    user_query      = state["messages"][-1].content if state["messages"] else ""
    kundali_details = state.get("kundali_details")
    user_profile    = state.get("user_profile")
    
    ##NOTE: IMPORTANT: If kundali_details or user_profile is not present, skip RAG and return False for needs_rag
    if not kundali_details or not user_profile:
        logger.warning("Missing kundali_details or user_profile, skipping RAG")
        state["needs_rag"] = False
        state["rag_query"] = None
        state["rag_context_keys"] = []
        state["metadata_filters"] = None
        return state
    
    #* Extract key astrological info
    sun_sign       = kundali_details.key_positions.sun.sign or "Unknown"
    moon_sign      = kundali_details.key_positions.moon.sign or "Unknown"
    ascendant_sign = kundali_details.key_positions.ascendant.sign or "Unknown"
    lagna_lord     = kundali_details.key_positions.lagna_lord or "Unknown"
    sun_nakshatra  = kundali_details.key_positions.sun.nakshatra or None
    moon_nakshatra = kundali_details.key_positions.moon.nakshatra or None
    
    #* Get planetary positions
    planets_info = []
    for planet in kundali_details.planets:
        planets_info.append(f"{planet.object}: {planet.rasi}")
    
    #* Check previous context from checkpointer (previous RAG results)
    previous_rag_results = state.get("rag_results", [])
    previous_context_summary = ""
    if previous_rag_results:
        previous_context_summary = "\n\nPrevious Context Available:\n"
        for i, result in enumerate(previous_rag_results[:3], 1):  # Show first 3 results
            content_preview = result.get("content", "")[:200]  # First 200 chars
            previous_context_summary += f"{i}. {content_preview}...\n"
        previous_context_summary += "\nIMPORTANT: Check if the user's current question can be answered using the previous context above. If yes, set needs_rag to False."
    
    #* Get previous context keys
    previous_context_keys = state.get("rag_context_keys", [])
    previous_keys_summary = ""
    if previous_context_keys:
        previous_keys_summary = f"\nPrevious Context Keys Used: {', '.join(previous_context_keys)}"
    
    #* LLM prompt with structured output
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""

You are an expert in Vedic astrology.
Your only aim is to analyze the preset context properly, understand the user's query properly and determine that to give the response,
whether RAG is needed or not. If RAG is needed, then determine the metadata filters and the rag query to use. 
 

Analyze the user's question, present context and their kundali details to determine:

1. If RAG (retrieval) is needed to answer the question
   - Check if the question can be answered with general astrological knowledge
   - Check if required information is already present in previous context (if provided)
   - Only set needs_rag=True if specific astrological information needs to be retrieved.
   - In case the grey area comes, use RAG to fetch the information.

2. What metadata filters to use (only if needs_rag=True):
   - zodiacs: CRITICAL - ONLY use the native's Sun Sign, Moon Sign, or Ascendant from the kundali. DO NOT use zodiacs from planetary positions. Only include zodiacs that are directly relevant to answering the question (e.g., if question is about personality, use sun sign; if about emotions, use moon sign; if about general life, use ascendant).
        - RULE: If question is about personality, use sun sign; if about emotions, use moon sign; if about general life, use ascendant.
   - planetary_factors: Filter by planets mentioned or relevant to the question, fetch proper planets info which can be used to reason on the question (e.g., Sun, Moon, Mars, Mercury, Jupiter, Venus, Saturn, Rahu, Ketu)
   - life_areas: Filter by life area (love, spirituality, career) if question is about that area
   - nakshtra: Filter by nakshatra if question is specifically about nakshatra (use Sun Nakshatra or Moon Nakshatra from kundali)

3. A search query optimized for semantic search (only if needs_rag=True)
   - Should be concise and focused on the specific astrological information needed
   - Include relevant astrological terms and concepts
   - Use keywords like: "traits", "characteristics", "personality", "career", "love", "relationships", "marriage", "spirituality", "guidance", "horoscope", "predictions", "compatibility", "strengths", "weaknesses", "remedies", "gemstones", "lucky", "unlucky"
   - Examples of good queries:
     * "Leo sun sign personality traits career guidance"
     * "Capricorn moon sign emotional characteristics love relationships"
     * "Aries ascendant life predictions career horoscope"
     * "Jupiter planetary influence career spirituality"
     * "Magha nakshatra traits marriage compatibility"

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

RAG INFO SUMMARY:
We have an extensive knowledge base consists of follwing type of information, and if user is asking for any of it or related, perform RAG to fetch the information.
- Personality traits of different zodiacs
- Career guidance for different zodiacs
- Love and relationships guidance for different zodiacs
- Spirituality guidance for different zodiacs
- Nakshatra traits for different zodiacs


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
    "rag_query"       : str | None,             # Optimized search query string for semantic search. Should be concise and focused (5-6 keywords) on the specific astrological information needed. Only provide if needs_rag is True. EXAMPLE: "Leo career and love relationships", "spiritual life taurus moon".
    "reasoning"       : str | None              # 30-40 words max reasoning for why RAG is needed or not needed, and what information is being sought. Only provide if needs_rag is True.

## IMPORTANT    
In reasoning, also write recommendations or hints or sutras which can help to solve the user query better.
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
        logger.error("\n\nQuery function not available in config\n\n")
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
        # Query ChromaDB with increased n_results for better retrieval
        # Increased from 3 to 10 to get more results, then filter by distance if needed
        results = query_function(
            query_text = rag_query,
            n_results  = 5,
            where      = where_clause
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
            
            # Process all retrieved documents (no distance filtering for now to get more results)
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
            
            # Limit to top 5 results
            rag_results = rag_results[:5]
        
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
    
    user_query       = state["messages"][-1].content if state["messages"] else ""
    kundali_details  = state.get("kundali_details")
    user_profile     = state.get("user_profile")
    rag_results      = state.get("rag_results", [])
    reasoning_summary        = state.get("reasoning", "")
    
    preferred_language = user_profile.preferred_language if user_profile else "en"
    language_name = "Hindi" if preferred_language == "hi" else "English"
    
    # Extract comprehensive kundali summary
    kundali_summary = ""
    if kundali_details:
        # Key Positions
        sun_sign        = kundali_details.key_positions.sun.sign or "Unknown"
        moon_sign       = kundali_details.key_positions.moon.sign or "Unknown"
        ascendant       = kundali_details.key_positions.ascendant.sign or "Unknown"
        lagna_lord      = kundali_details.key_positions.lagna_lord or "Unknown"
        
        sun_nakshatra   = kundali_details.key_positions.sun.nakshatra or "Unknown"
        moon_nakshatra  = kundali_details.key_positions.moon.nakshatra or "Unknown"
        sun_nakshatra_lord = kundali_details.key_positions.sun.nakshatra_lord or "Unknown"
        moon_nakshatra_lord = kundali_details.key_positions.moon.nakshatra_lord or "Unknown"
        
        # Build key positions summary
        kundali_summary = f"""Key Positions:
- Sun: {sun_sign} (Nakshatra: {sun_nakshatra}, Nakshatra Lord: {sun_nakshatra_lord})
- Moon: {moon_sign} (Nakshatra: {moon_nakshatra}, Nakshatra Lord: {moon_nakshatra_lord})
- Ascendant (Lagna): {ascendant} (Lagna Lord: {lagna_lord})
"""
        
        # Planetary Positions
        if kundali_details.planets:
            kundali_summary += "\nPlanetary Positions:\n"
            for planet in kundali_details.planets:
                planet_name = planet.object
                planet_sign = planet.rasi or "Unknown"
                planet_house = planet.house_nr if planet.house_nr is not None else "Unknown"
                planet_nakshatra = planet.nakshatra or "Unknown"
                is_retrograde = " (Retrograde)" if planet.is_retrograde else ""
                
                kundali_summary += f"- {planet_name}: {planet_sign} in House {planet_house}"
                if planet_nakshatra != "Unknown":
                    kundali_summary += f" (Nakshatra: {planet_nakshatra})"
                kundali_summary += f"{is_retrograde}\n"
        
        # Current Dasa Information
        if kundali_details.vimshottari_dasa:
            logger.debug(f"\n\nVimshottari Dasa Information: {kundali_details.vimshottari_dasa}\n\n")
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
                
                current_date = datetime.now().date()
                current_dasa_name = None
                current_dasa_data = None
                
                # Find the current dasa (the one that contains today's date)
                for dasa_name, dasa_info in kundali_details.vimshottari_dasa.items():
                    try:
                        dasa_start = parse_dasha_date(dasa_info.start)
                        dasa_end = parse_dasha_date(dasa_info.end)
                        
                        if dasa_start <= current_date <= dasa_end:
                            current_dasa_name = dasa_name
                            current_dasa_data = dasa_info
                            break
                    except (ValueError, AttributeError) as e:
                        logger.debug(f"Error parsing dasa dates for {dasa_name}: {e}")
                        continue
                
                if current_dasa_name and current_dasa_data:
                    kundali_summary += "\nCurrent Vimshottari Dasa Period:\n"
                    kundali_summary += f"- {current_dasa_name}: {current_dasa_data.start} to {current_dasa_data.end}\n"
                    
                    # Find and show current bhukti
                    if current_dasa_data.bhuktis:
                        for bhukti_name, bhukti_info in current_dasa_data.bhuktis.items():
                            try:
                                bhukti_start = parse_dasha_date(bhukti_info.start)
                                bhukti_end = parse_dasha_date(bhukti_info.end)
                                
                                if bhukti_start <= current_date <= bhukti_end:
                                    kundali_summary += f"  - Current Bhukti: {bhukti_name} ({bhukti_info.start} to {bhukti_info.end})\n"
                                    break
                            except (ValueError, AttributeError) as e:
                                logger.debug(f"Error parsing bhukti dates for {bhukti_name}: {e}")
                                continue
            except Exception as e:
                logger.warning(f"Error extracting dasha info in nodes: {str(e)}", exc_info=True)
                # If error occurs, skip dasha info
                pass
        
        # Important Planetary Aspects (if available)
        if kundali_details.planetary_aspects:
            kundali_summary += "\nPlanetary Aspects:\n"
            for aspect in kundali_details.planetary_aspects[:5]:  # Show first 5 aspects
                kundali_summary += f"- {aspect.P1} aspects {aspect.P2} ({aspect.AspectType}, {aspect.AspectDeg}°)\n"
    
    # Prepare RAG context
    rag_context = ""
    if rag_results:
        rag_context = "\n\nRelevant Astrological Information:\n"
        for i, result in enumerate(rag_results, 1):
            rag_context += f"{i}. {result['content']}\n"
    
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
- Keep responses concise: Aim for 30-50 words maximum. Be direct, optimistic, real, and impactful.
- Maintain an optimistic and encouraging tone: Focus on positive aspects, growth opportunities, and constructive guidance
- Use proper, respectful language: Address the user with dignity and maintain professional boundaries
- Reference specific astrological placements: Mention relevant planets, signs, houses, or nakshatras when applicable
- Provide actionable insights: Offer practical guidance that users can apply to their lives
- Stay within astrological scope: Focus on astrological interpretations and avoid medical, legal, or financial advice
- Do not use complex terms or jargon, keep it simple and easy to understand.

Handling Time-Based Queries (Current Transits, Today, This Week, This Month):
When the user asks questions about current time periods like "why is today stressful", "what should I focus on this month", "why am I feeling this way", "what's happening to me right now", or similar time-sensitive questions:

1. Identify Time-Based Queries:
   - Look for keywords: "today", "this week", "this month", "right now", "current", "why is", "what should", "feeling", "stressful", "difficult", "challenging", "focus", "should I"
   - These questions require considering current planetary influences

2. What to Consider:
   - Current Dasha Period: Check the current dasa period in the kundali details (the one that contains today's date)
   - Current Bhukti: Check the current bhukti within the current dasa (the one that contains today's date)
   - Planetary Positions: Consider which planets are strong based on the current dasa/bhukti
   - Birth Chart Strengths: Reference the user's planetary positions from their birth chart
   - House Positions: Consider which life areas (houses) are being activated by the current period

3. How to Answer:
   - Explain in simple, everyday language without technical terms
   - Instead of "Saturn transit in 7th house", say "Saturn's energy is currently affecting your relationships area"
   - Instead of "Mars dasha", say "Mars energy is strong in your current period"
   - Instead of "retrograde", say "this planet is moving slowly"
   - Connect the current period to their birth chart: "Because your birth chart shows [planet] in [sign], the current [dasa] period is bringing [effect]"
   - For "why stressful": Explain which planetary energies are challenging based on current period + birth chart
   - For "what to focus on": Suggest areas of life that are favorable based on current period + birth chart strengths
   - Be empathetic and practical: "You might be feeling [way] because [simple explanation]. Try focusing on [practical advice]"

4. Example Response Structure:
   - Acknowledge their feeling/question
   - Explain which planetary influences are active (in simple terms)
   - Connect to their birth chart strengths/challenges
   - Provide practical, actionable guidance
   - Keep it optimistic and empowering

Handling Compatibility and Relationship Queries:
When the user asks questions about relationships, love, marriage, compatibility, or partnership like "will I find love", "when will I get married", "is this person right for me", "why am I single", "relationship problems", or similar questions:

1. Identify Relationship Queries:
   - Look for keywords: "love", "marriage", "relationship", "partner", "compatibility", "single", "dating", "romance", "spouse", "husband", "wife", "divorce", "breakup", "when will I", "will I find"
   - These questions focus on the 7th house (partnerships) and Venus (love), Mars (passion), Jupiter (marriage)

2. What to Consider:
   - 7th House: Check which sign and planet rules the 7th house (partnerships/marriage)
   - Venus Position: Venus represents love, relationships, and harmony - check its sign, house, and aspects
   - Mars Position: Mars represents passion and attraction - check its influence
   - Jupiter Position: Jupiter represents marriage and expansion - check its role
   - Current Dasha: Consider if current dasha period favors relationships (Venus, Jupiter, or 7th house lord)
   - Moon Sign: Emotional needs and how they relate to relationships
   - Planetary Aspects: Check if planets are supporting or challenging relationship matters

3. How to Answer:
   - Explain in simple, empathetic language without technical terms
   - Instead of "7th house lord in 12th", say "your chart shows that partnerships might require some adjustments or spiritual growth"
   - Instead of "Venus debilitated", say "Venus energy needs extra care in relationships"
   - Connect planetary positions to relationship patterns: "Because your chart shows [planet] in [sign], you tend to [relationship pattern]"
   - For timing questions: Reference current dasha/bhukti periods that favor relationships
   - For compatibility: Explain what signs/energies complement their chart
   - Be encouraging: "Your chart shows [positive aspect]. Focus on [practical advice] to attract the right partner"
   - Avoid making definitive predictions about specific people or exact dates

4. Example Response Structure:
   - Acknowledge their question/concern
   - Explain relevant planetary influences in simple terms
   - Connect to their birth chart's relationship indicators
   - Provide guidance on what to focus on or work on
   - Keep it hopeful and practical

Handling Career and Professional Guidance Queries:
When the user asks questions about career, work, profession, business, or professional growth like "what career suits me", "should I change jobs", "will I be successful", "when will I get promotion", "business opportunities", or similar questions:

1. Identify Career Queries:
   - Look for keywords: "career", "job", "profession", "work", "business", "success", "promotion", "salary", "career change", "what should I do", "which field", "opportunities"
   - These questions focus on the 10th house (career/profession) and relevant planets

2. What to Consider:
   - 10th House: Check which sign and planet rules the 10th house (career/profession)
   - Sun Position: Sun represents authority, leadership, and career - check its sign, house, and strength
   - Mercury Position: Mercury represents communication, business, and intellect - important for career
   - Saturn Position: Saturn represents discipline, structure, and long-term career - check its influence
   - Current Dasha: Consider if current dasha period favors career growth (10th house lord, Sun, Mercury, Saturn)
   - Planetary Aspects: Check which planets are supporting or challenging career matters
   - House Positions: Consider which houses are strong (indicates natural talents and career inclinations)

3. How to Answer:
   - Explain in simple, practical language without technical terms
   - Instead of "10th house lord in 5th", say "your career is connected to creative or educational fields"
   - Instead of "Saturn aspecting Sun", say "discipline and hard work will be important for your career growth"
   - Connect planetary positions to career inclinations: "Because your chart shows [planet] in [sign/house], you have natural talent for [field]"
   - For career choice: Suggest fields based on strong planets and houses in their chart
   - For timing questions: Reference current dasha/bhukti periods that favor career opportunities
   - Be practical: "Your chart shows strength in [area]. Consider focusing on [career path] and developing [skills]"
   - Encourage growth: "This is a good time to [action] because [planetary influence explanation]"

4. Example Response Structure:
   - Acknowledge their career question
   - Explain relevant planetary influences in simple terms
   - Connect to their birth chart's career indicators and natural talents
   - Provide practical guidance on career direction or actions to take
   - Keep it encouraging and actionable
   

## Some Sutras
That's a great approach—creating a simple, actionable list of focal points for an astrological query.

Here is a concise prompt you can copy and use, where each point defines the essential component to focus on and briefly explains how to analyze it.

---

## 📋 Astrological Analysis Focus Points Prompt

Please provide a detailed predictive analysis focusing on the following 12 key astrological components:

1.  **Current Dasha & Antardasha:** Identify the ruling Major (Dasha) and Sub (Antardasha) Lords, and analyze the **houses they rule and occupy** in the birth chart to determine the current major life themes and priorities.
2.  **Saturn's (Shani) Current Transit:** Analyze Saturn's current position relative to the **Natal Moon** (Sade Sati, Ashtam Shani, etc.) to assess the nature and duration of current karma, discipline, and restriction.
3.  **Jupiter's (Guru) Current Transit:** Analyze Jupiter's current position relative to the **Natal Moon** and **Lagna (Ascendant)** to predict areas of expansion, luck, opportunity, and beneficial growth.
4.  **Rahu-Ketu Transit Axis:** Identify the houses currently occupied by the Nodal Axis to determine the **primary focus of intense desire and detachment** in the native's life for this 18-month period.
5.  **Transit to Natal Position:** Check for any slow-moving planets (Saturn, Jupiter, Rahu, Ketu) transiting over their **natal placement** to identify activation points of past life lessons or karmic events.
6.  **Functional Benefics/Malefics (Yogas):** Identify the planets that act as natural benefics (good) and malefics (bad) based on the **Lagna/Ascendant** to determine the overall quality of the Dasha and Transit results.
7.  **Yogakaraka Planet:** Identify the single most beneficial planet that rules both a **Kendra (Angular)** and a **Trikona (Trine) house**. Analyze its Dasha and Transit influence for peak fortune.
8.  **Dasha Lord House Position:** Analyze the **house occupied by the Dasha Lord** in the birth chart (Rashi Chart) to define the most significant area of life being activated during the entire Major Period.
9.  **Sub-Lord (Antardasha Lord) Strength:** Assess the strength and dignity of the current Antardasha Lord to determine the **intensity and ease** with which events promised by the Major Dasha will manifest.
10. **Retrograde (Vakra) Influence:** Identify any planets that are currently **retrograde** in transit (Gochar) to note potential **delays, repetitions, or internal revisions** related to the houses they rule and aspect.   
11. Use current transits, native's birthchart, the houses, the rashis of each house, the planets sitting and different combinations to reason properly   

User's Birth Kundali Details:
{kundali_summary}

Above is user's birth kundali details. You should consider this as well as current month/year transits to form your reasoning and response.

## THIS IS THE CONTEXT FROM RAG RESULTS, use this to provide insights better.
{rag_context}


## Reasoning Summary:
{reasoning_summary}

IMPORTANT: You MUST respond strictly in {language_name} ({preferred_language}).


EXAMPLE COVERSATIONS:

--1--
User: I feel stuck in my job. Should I switch fields?
Response: Currently, you are in the Mars (Mangal) Mahadasha with Rahu Antardasha. Mars rules your 10th house of career and is placed in your 3rd house of courage. This energy suggests a strong drive for independent action, but Rahu can create confusion or a desire for sudden change. Instead of a complete field switch, look for leadership roles or competitive projects within your current industry. The upcoming transit of Jupiter into your 9th house next month will bring clarity and new opportunities. Trust your initiative, but avoid impulsive decisions right now.

--2--
User: When will I meet someone special? I've been single for a long time.
Response: Your chart shows Venus, the planet of love, is well-placed in Pisces, indicating a deep capacity for connection. Currently, Saturn is transiting your 7th house of partnerships, which often delays relationships to teach patience and maturity. However, with Jupiter aspecting your 5th house of romance starting in May, the energy shifts to become more favorable for meeting someone new. Focus on building your own stability during this time; a meaningful connection is likely to arrive when you are most grounded, specifically during the Venus Antardasha starting late this year.

--3--
User: Why is everything so difficult right now? I feel mentally drained.
Response: You are currently in the peak phase of Sade Sati, where Saturn is transiting over your natal Moon. This period often feels heavy because Saturn forces us to confront our deepest emotional insecurities and restructure our lives. Additionally, the Moon is currently impacted by a Ketu transit, adding to a sense of detachment or anxiety. This is not permanent; it is a time for introspection and spiritual growth. Prioritize rest and routine. As Saturn moves forward in six months, this mental pressure will lift significantly.

--4--
User: What is my main purpose in life?
Response: Your Lagna is Leo, ruled by the Sun, giving you a natural radiance and a need for self-expression. Your Yogakaraka planet, Mars, is in the 5th house, suggesting your purpose is tied to creativity, intellect, or mentoring others. You thrive when you lead with your heart and take calculated risks. Your chart indicates fulfillment comes not just from status, but from creating something that leaves a legacy. Embrace roles where you can inspire others; that is where your true strength lies.

--5--
User: पैसों की बहुत दिक्कत चल रही है। कब ठीक होगा?
Response: अभी आपकी गुरु (Jupiter) की महादशा चल रही है, जो आपके दूसरे भाव (धन भाव) के स्वामी हैं। हालांकि, गोचर में शनि की दृष्टि आपके ग्यारहवें भाव (लाभ भाव) पर है, जो आय (income) में कुछ रुकावट डाल रहा है। यह समय धैर्य रखने और बजट को संभालने का है। आने वाले नवंबर के बाद, जब शनि मार्गी होंगे, तब आर्थिक स्थिति में सुधार दिखेगा। फिलहाल नए निवेश से बचें और बचत पर ध्यान दें। चिंता न करें, यह स्थिति स्थायी नहीं है।

--6--
User: मेरी शादी में काफी झगड़े हो रहे हैं। क्या करें?
Response: आपकी कुंडली में मंगल (Mars) सातवें भाव में है, जो विवाह में थोड़ा क्रोध और जल्दबाजी (impulsiveness) लाता है। इस समय राहु का गोचर भी आपके लग्न पर है, जिससे बेचैनी और भ्रम बढ़ सकता है। आपके लिए ज़रूरी है कि आप शांत रहें और प्रतिक्रिया देने से पहले सोचें। शुक्र (Venus) आपके चार्ट में मजबूत है, जो प्रेम और सामंजस्य बनाए रखेगा। अपने जीवनसाथी की बात सुनने का प्रयास करें; शुक्रवार को सफेद वस्त्र धारण करने से संबंधों में शांति मिलेगी।

--7--
User: तबीयत बार-बार खराब हो रही है आजकल।
Response: आपकी शनि की महादशा और केतु की अंतर्दशा चल रही है। केतु छठे भाव (रोग भाव) में होने के कारण, रोग के निदान (diagnosis) में कुछ परेशानी या छोटी-मोटी समस्याएँ आ सकती हैं। यह समय शारीरिक से अधिक मानसिक तनाव का परिणाम हो सकता है। योग और ध्यान (meditation) को अपनी दिनचर्या में शामिल करें। सूर्यदेव को जल अर्पित करें, क्योंकि सूर्य आपके लग्नेश हैं और रोग प्रतिरोधक क्षमता (immunity) बढ़ाएंगे। घबराएँ नहीं, यह समय स्वास्थ्य के प्रति अधिक सावधानी बरतने का है।



IMPORTANT: You MUST respond strictly in {language_name} ({preferred_language}) in very polite, optimistic, encouraging, and empowering tone.

Guidelines for Handling Inappropriate Queries:
- If the user's question contains NSFW content, dangerous requests, harmful instructions, or topics unrelated to astrology, respond politely but firmly
- Decline requests for illegal activities, self-harm guidance, violence, or explicit adult content with a professional boundary statement
- For non-astrology related queries, politely redirect: "I specialize in Vedic astrology guidance. How can I help you understand your kundali better?"
- Maintain ethical standards: Do not provide guidance that could cause harm or encourage dangerous behavior
- If a query is inappropriate, respond formally: "I'm here to provide astrological guidance based on your kundali. I cannot assist with that request, but I'm happy to help with astrology-related questions."
"""),
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
