"""
Simple logging utility for data ingestion and processing.

Provides a configured logger with INFO level by default.
"""

import logging
import sys
from pathlib import Path


def setup_logger(name: str = "data_ingestion", level: int = logging.INFO) -> logging.Logger:
    """
    Set up and configure a logger with console output.

    Args:
        name (str): Logger name
        level (int): Logging level (default: INFO)

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding multiple handlers if logger already exists
    if logger.handlers:
        return logger

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(handler)

    return logger


# Create default logger instance
logger = setup_logger()

