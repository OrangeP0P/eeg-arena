"""
Logging utilities for experiment tracking
"""
import os
import json
import logging
from datetime import datetime


def setup_logger(name, log_dir=None, level=logging.INFO):
    """
    Setup logger with file and console handlers

    Args:
        name: logger name
        log_dir: directory to save log files
        level: logging level

    Returns:
        logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)

    return logger


def save_config(config, save_dir, filename='config.json'):
    """
    Save configuration to JSON file

    Args:
        config: dict or argparse.Namespace
        save_dir: directory to save
        filename: filename
    """
    os.makedirs(save_dir, exist_ok=True)

    if hasattr(config, '__dict__'):
        config = vars(config)

    save_path = os.path.join(save_dir, filename)
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)

    return save_path


def load_config(config_path):
    """
    Load configuration from JSON file

    Args:
        config_path: path to config file

 dict
    Returns:
           """
    with open(config_path, 'r') as f:
        return json.load(f)


__all__ = ['setup_logger', 'save_config', 'load_config']
