import os
import logging
import yaml
import re
import hashlib
from typing import Dict, Any, Optional


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> None:
    """
    Set up logging configuration.

    Args:
        log_file: Path to log file (optional)
        level: Logging level
    """
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Create handlers
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    # Configure logging
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=handlers
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Dictionary containing configuration
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def clean_text(text: str) -> str:
    """
    Clean and normalize text.

    Args:
        text: Input text

    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,;:?!()\'"-]', '', text)

    return text


def create_unique_id(text: str) -> str:
    """
    Create a unique ID for a piece of text.

    Args:
        text: Input text

    Returns:
        MD5 hash of the text
    """
    return hashlib.md5(text.encode()).hexdigest()
