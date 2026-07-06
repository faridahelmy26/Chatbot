import logging
import sys
from datetime import datetime
from typing import Optional

from config import LOG_LEVEL


def setup_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Setup logger for the application
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Logger instance
    """
    if level is None:
        level = LOG_LEVEL
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


class Logger:
    """Custom logger wrapper"""
    
    def __init__(self, name: str):
        self.logger = setup_logger(name)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def debug(self, message: str):
        self.logger.debug(message)
    
    def critical(self, message: str):
        self.logger.critical(message)
