"""
File processing utilities for JSON and text files.
"""

import json
import uuid
from typing import List, Dict, Any
from pathlib import Path
from .metadata import create_metadata
from .logger import logger


def process_json_file(file_path: str, collection) -> None:
    """
    Process JSON files by chunking each key-value pair within main objects.

    Args:
        file_path (str): Path to the JSON file
        collection: Chroma collection to add documents to
    """
    filename = Path(file_path).stem
    logger.info(f"Processing JSON file: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.debug(f"Loaded JSON data from {filename}, found {len(data)} top-level keys")
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {file_path}: {e}")
        raise

    documents = []
    metadatas = []
    ids = []

    for main_key, main_value in data.items():
        if isinstance(main_value, dict):
            # Process each key-value pair within the main object
            for sub_key, sub_value in main_value.items():
                if isinstance(sub_value, list):
                    sub_value_str = ", ".join(str(item) for item in sub_value)
                else:
                    sub_value_str = str(sub_value)

                # Create document combining key and value
                document = f"{sub_key}: {sub_value_str}"

                # Create metadata and unique ID
                metadata = create_metadata(filename, main_key)
                doc_id = f"{filename}_{main_key}_{sub_key}_{str(uuid.uuid4())}".replace(" ", "_").lower()

                documents.append(document)
                metadatas.append(metadata)
                ids.append(doc_id)

    if documents:
        logger.info(f"Adding {len(documents)} documents from {filename} to collection...")
        try:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"✓ Successfully added {len(documents)} documents from {filename}")
        except Exception as e:
            logger.error(f"Failed to add documents from {filename}: {e}")
            raise
    else:
        logger.warning(f"No documents extracted from {filename}")


def process_text_file(file_path: str, collection) -> None:
    """
    Process text files by splitting on sentences (bullet points).

    Args:
        file_path (str): Path to the text file
        collection: Chroma collection to add documents to
    """
    filename = Path(file_path).stem
    logger.info(f"Processing text file: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        logger.debug(f"Loaded text file {filename}, content length: {len(content)} characters")
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise

    # Split by lines and filter out empty lines
    sentences = [line.strip().lstrip('-').strip() for line in content.split('\n') if line.strip()]
    logger.debug(f"Extracted {len(sentences)} sentences from {filename}")

    documents = []
    metadatas = []
    ids = []

    for i, sentence in enumerate(sentences):
        if sentence:  # Only add non-empty sentences
            metadata = create_metadata(filename)
            doc_id = f"{filename}_sentence_{i+1}_{str(uuid.uuid4())}".replace(" ", "_").lower()

            documents.append(sentence)
            metadatas.append(metadata)
            ids.append(doc_id)

    if documents:
        logger.info(f"Adding {len(documents)} documents from {filename} to collection...")
        try:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"✓ Successfully added {len(documents)} documents from {filename}")
        except Exception as e:
            logger.error(f"Failed to add documents from {filename}: {e}")
            raise
    else:
        logger.warning(f"No documents extracted from {filename}")

