"""
Simple script to run data ingestion into vector database.

Usage:
    python -m helper.run_insert
    or
    python helper/run_insert.py
"""

import sys
from pathlib import Path

# Add parent directory to path for absolute imports when run directly
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from helper.data_ingestion import ingest_data


if __name__ == "__main__":
    ingest_data()

