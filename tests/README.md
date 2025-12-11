# Test Suite Documentation

This directory contains comprehensive tests for the MyNakshpoc application, organized by component.

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and mocks
├── app/                     # Tests for app components
│   ├── test_models.py       # Pydantic model tests
│   ├── test_llmclient.py    # LLM client tests
│   ├── test_state.py        # GraphState tests
│   ├── test_builder.py      # Graph builder tests
│   ├── test_utils.py        # Utility function tests
│   ├── test_nodes.py        # LangGraph node tests
│   ├── test_router_chat.py  # Chat router tests
│   └── test_router_kundali.py # Kundali router tests
└── helper/                  # Tests for helper components
    ├── test_data_ingestion.py    # Data ingestion tests
    ├── test_init_chroma_db.py    # ChromaDB initialization tests
    ├── test_run_insert.py        # Run insert script tests
    └── utils/                    # Helper utility tests
        ├── test_embeddings.py    # Embedding function tests
        ├── test_file_processors.py # File processor tests
        ├── test_metadata.py       # Metadata creation tests
        └── test_logger.py         # Logger utility tests
```

## Running Tests

### Install Dependencies
```bash
uv sync --extra dev
# or
pip install -e ".[dev]"
```

### Run All Tests
```bash
pytest
```

### Run Specific Test File
```bash
pytest tests/app/test_models.py
```

### Run with Coverage
```bash
pytest --cov=app --cov=helper --cov-report=html
```

### Run with Verbose Output
```bash
pytest -v
```

## Test Documentation

Each test file includes comprehensive docstrings explaining:
- **What**: What is being tested
- **Why**: Why the test is important
- **Args**: What arguments/inputs are used

Each test class and method follows this documentation pattern.

## Test Coverage

### App Components
- ✅ **Models**: Pydantic model validation, field types, validators
- ✅ **LLM Client**: Model configuration, temperature settings
- ✅ **State**: GraphState structure and field types
- ✅ **Builder**: Graph construction and compilation
- ✅ **Utils**: Geocoding, timezone, datetime parsing, kundali calculation
- ✅ **Nodes**: RAG query generation, retrieval, chat response generation
- ✅ **Routers**: Chat and kundali endpoint testing

### Helper Components
- ✅ **Data Ingestion**: File processing, collection initialization
- ✅ **ChromaDB**: Collection creation, loading, query functions
- ✅ **Embeddings**: OpenAI embedding function creation and configuration
- ✅ **File Processors**: JSON and text file processing
- ✅ **Metadata**: Metadata creation for different content types
- ✅ **Logger**: Logger configuration and handler setup

## Fixtures

Shared fixtures are defined in `conftest.py`:
- `mock_user_profile`: Mock UserProfile for testing
- `mock_kundali_details`: Mock KundaliDetails with test astrological data
- `mock_graph_state`: Mock GraphState for LangGraph tests
- `mock_chroma_collection`: Mock ChromaDB collection
- `mock_query_function`: Mock query function for ChromaDB
- `mock_geocoder`: Mock geocoder for location services
- `mock_vedic_data`: Mock VedicHoroscopeData
- `mock_fastapi_request`: Mock FastAPI Request object

## Mocking Strategy

Tests use `unittest.mock` and `pytest-mock` for:
- External API calls (OpenAI, geocoding)
- Database operations (ChromaDB)
- File I/O operations
- Async operations (pytest-asyncio)

## Best Practices

1. **Isolation**: Each test is independent and doesn't rely on other tests
2. **Mocking**: External dependencies are mocked to ensure fast, reliable tests
3. **Documentation**: All tests include docstrings explaining purpose
4. **Coverage**: Tests cover happy paths, error cases, and edge cases
5. **Async Support**: Async functions are tested with `pytest-asyncio`

## Notes

- Tests use `pytest-asyncio` for async function testing
- Mock objects are used extensively to avoid external dependencies
- Test data is minimal but representative of real usage
- Error handling is tested to ensure proper exception propagation

