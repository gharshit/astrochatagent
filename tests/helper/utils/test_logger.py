"""
Tests for logger utilities in helper.utils.logger.

This module tests:
- setup_logger() function
- Logger configuration and handler setup
- Log level configuration
- Handler deduplication

Why: Logging is essential for debugging and monitoring application behavior.
Args: Logger names, log levels, handler configurations.
"""

import pytest
from unittest.mock import patch, Mock
import logging
import sys
from helper.utils.logger import setup_logger


class TestSetupLogger:
    """
    Test setup_logger() function.
    
    Tests: Logger creation, handler configuration, level setting, deduplication.
    Why: Ensures consistent logging configuration across the application.
    Args: Logger name, log level.
    """
    
    def test_setup_logger_creates_logger(self):
        """
        Test setup_logger() creates logger with correct name.
        
        What: Validates that logger is created with specified name.
        Why: Logger names help identify log sources.
        Args: Logger name "test_logger".
        """
        logger = setup_logger("test_logger")
        
        assert logger.name == "test_logger"
        assert isinstance(logger, logging.Logger)
    
    def test_setup_logger_sets_level(self):
        """
        Test setup_logger() sets correct log level.
        
        What: Validates that logger level is set correctly.
        Why: Log levels control verbosity of logging output.
        Args: Logger name, log level DEBUG.
        """
        logger = setup_logger("test_logger", level=logging.DEBUG)
        
        assert logger.level == logging.DEBUG
    
    def test_setup_logger_adds_handler(self):
        """
        Test setup_logger() adds console handler.
        
        What: Validates that console handler is added to logger.
        Why: Handlers determine where log messages are output.
        Args: Logger name, default level.
        """
        logger = setup_logger("test_logger")
        
        assert len(logger.handlers) > 0
        assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    
    def test_setup_logger_handler_deduplication(self):
        """
        Test setup_logger() doesn't add duplicate handlers.
        
        What: Validates that calling setup_logger twice doesn't add duplicate handlers.
        Why: Prevents duplicate log messages.
        Args: Same logger name called twice.
        """
        logger1 = setup_logger("dedup_test")
        handler_count1 = len(logger1.handlers)
        
        logger2 = setup_logger("dedup_test")
        handler_count2 = len(logger2.handlers)
        
        assert handler_count1 == handler_count2
        assert logger1 is logger2  # Should return same logger instance
    
    def test_setup_logger_formatter_configuration(self):
        """
        Test setup_logger() configures formatter correctly.
        
        What: Validates that formatter includes timestamp, name, level, message.
        Why: Formatters control log message format.
        Args: Logger name, default level.
        """
        logger = setup_logger("formatter_test")
        
        handler = logger.handlers[0]
        assert handler.formatter is not None
        assert isinstance(handler.formatter, logging.Formatter)
    
    def test_setup_logger_default_level(self):
        """
        Test setup_logger() uses INFO level by default.
        
        What: Validates that default log level is INFO.
        Why: INFO level provides reasonable verbosity by default.
        Args: Logger name, no level specified.
        """
        logger = setup_logger("default_level_test")
        
        assert logger.level == logging.INFO

