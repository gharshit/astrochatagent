# MyNakshpoc

A Vedic astrology knowledge base using ChromaDB with OpenAI embeddings via LangChain.

## Setup

1. Install dependencies:
```bash
uv sync
```

2. Create a `.env` file in the root directory with your OpenAI API key and model configuration:
```env
OPENAI_API_KEY=your_openai_api_key_here

# LLM Model Configuration (optional - defaults provided)
LLM_CHAT_MODEL=gpt-4o-mini
LLM_STRUCTURED_MODEL=gpt-4o-mini
LLM_CHAT_TEMPERATURE=0.7
LLM_STRUCTURED_TEMPERATURE=0
LLM_EMBEDDING_MODEL=text-embedding-3-small
```

3. Run data ingestion:
```bash
cd db
uv run python data_ingestion.py
```

## Project Structure

- `db/` - Database initialization and data ingestion scripts
- `data/` - Source data files (JSON and text files)
- `vector_db/` - ChromaDB persistent storage (created automatically)

## Features

- Uses LangChain with OpenAI embeddings for semantic search
- Processes JSON files by chunking key-value pairs
- Processes text files by splitting on sentences
- Rich metadata support for filtering (zodiacs, planetary_factors, life_areas)

