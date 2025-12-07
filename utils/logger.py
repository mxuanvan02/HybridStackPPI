#!/usr/bin/env python3
"""
Enhanced Logger for HybridStackPPI
==================================
Colored console logging with file output and tqdm integration.

Features:
- Dual output: Console (colored) + File (plain)
- Integration with tqdm progress bars (no output corruption)
- Structured logging levels (INFO, WARNING, ERROR, SUCCESS)
- Phase markers for experiment organization
"""

import logging
import sys
from pathlib import Path
from typing import Optional

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False
    # Fallback: no colors
    class Fore:
        GREEN = RED = YELLOW = CYAN = BLUE = MAGENTA = ""
    class Style:
        BRIGHT = RESET_ALL = ""

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class EnhancedLogger:
    """
    Logger with colored output, file logging, and tqdm support.
    
    Example:
        >>> logger = EnhancedLogger("experiment", log_file="results/exp.log")
        >>> logger.info("Starting training...")
        >>> logger.success("Training complete!")
        >>> logger.warning("Low memory detected")
        >>> logger.error("Failed to load data")
        
        >>> # With tqdm
        >>> for epoch in logger.tqdm(range(10), desc="Training"):
        ...     logger.info(f"Epoch {epoch}")
    """
    
    def __init__(
        self,
        name: str = "HybridStackPPI",
        log_file: Optional[str] = None,
        level: int = logging.INFO,
        console: bool = True,
    ):
        """
        Initialize enhanced logger.
        
        Args:
            name: Logger name
            log_file: Path to log file (optional)
            level: Logging level
            console: Whether to output to console
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers.clear()  # Remove existing handlers
        
        # File handler (plain text)
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file, mode='a')
            file_handler.setLevel(level)
            file_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        
        # Console handler (with colors)
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            
            # Custom formatter for colors
            class ColoredFormatter(logging.Formatter):
                COLORS = {
                    'DEBUG': Fore.CYAN,
                    'INFO': Fore.BLUE,
                    'WARNING': Fore.YELLOW,
                    'ERROR': Fore.RED,
                    'CRITICAL': Fore.RED + Style.BRIGHT,
                }
                
                def format(self, record):
                    if COLORAMA_AVAILABLE:
                        color = self.COLORS.get(record.levelname, '')
                        record.levelname = f"{color}{record.levelname}{Style.RESET_ALL}"
                        record.msg = f"{color}{record.msg}{Style.RESET_ALL}"
                    return super().format(record)
            
            colored_formatter = ColoredFormatter(
                '%(levelname)s | %(message)s'
            )
            console_handler.setFormatter(colored_formatter)
            self.logger.addHandler(console_handler)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def success(self, message: str):
        """Log success message (custom level)."""
        if COLORAMA_AVAILABLE:
            formatted = f"{Fore.GREEN}✅ {message}{Style.RESET_ALL}"
        else:
            formatted = f"✅ {message}"
        self.logger.info(formatted)
    
    def header(self, message: str, char: str = "="):
        """
        Log header with decorations.
        
        Args:
            message: Header text
            char: Decoration character
        """
        line = char * 70
        if COLORAMA_AVAILABLE:
            formatted = f"\n{Fore.MAGENTA}{Style.BRIGHT}{line}\n{message}\n{line}{Style.RESET_ALL}"
        else:
            formatted = f"\n{line}\n{message}\n{line}"
        self.logger.info(formatted)
    
    def phase(self, message: str):
        """
        Log experiment phase marker.
        
        Args:
            message: Phase description
        """
        if COLORAMA_AVAILABLE:
            formatted = f"\n{Fore.CYAN}{Style.BRIGHT}🔹 PHASE: {message}{Style.RESET_ALL}"
        else:
            formatted = f"\n🔹 PHASE: {message}"
        self.logger.info(formatted)
    
    def metric(self, name: str, value: float, format_spec: str = ".4f"):
        """
        Log metric value.
        
        Args:
            name: Metric name
            value: Metric value
            format_spec: Format specification
        """
        if COLORAMA_AVAILABLE:
            formatted = f"{Fore.GREEN}{name:<25} {value:{format_spec}}{Style.RESET_ALL}"
        else:
            formatted = f"{name:<25} {value:{format_spec}}"
        self.logger.info(formatted)
    
    def tqdm(self, iterable, desc: str = "", **kwargs):
        """
        Create tqdm progress bar with logger integration.
        
        Args:
            iterable: Iterable to wrap
            desc: Description
            **kwargs: Additional tqdm arguments
        
        Returns:
            tqdm iterator or plain iterator if tqdm unavailable
        """
        if TQDM_AVAILABLE:
            # Use tqdm with file parameter to avoid conflicts
            return tqdm(
                iterable,
                desc=desc,
                file=sys.stdout,
                ncols=80,
                **kwargs
            )
        else:
            # Fallback: plain iteration with logging
            self.info(f"Starting: {desc}")
            return iterable
    
    def progress(self, current: int, total: int, prefix: str = "Progress"):
        """
        Log simple progress indicator.
        
        Args:
            current: Current step
            total: Total steps
            prefix: Progress prefix
        """
        percentage = (current / total) * 100
        bar_length = 40
        filled = int(bar_length * current / total)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        message = f"{prefix}: [{bar}] {current}/{total} ({percentage:.1f}%)"
        self.info(message)
    
    def table(self, data: dict, title: Optional[str] = None):
        """
        Log data as formatted table.
        
        Args:
            data: Dictionary of key-value pairs
            title: Optional table title
        """
        if title:
            self.info(f"\n{title}")
            self.info("-" * 70)
        
        max_key_len = max(len(str(k)) for k in data.keys())
        
        for key, value in data.items():
            self.info(f"  {str(key):<{max_key_len}} : {value}")
        
        if title:
            self.info("-" * 70)
    
    def separator(self, char: str = "-", length: int = 70):
        """
        Log separator line.
        
        Args:
            char: Separator character
            length: Line length
        """
        self.info(char * length)


# Convenience function
def create_logger(
    name: str = "HybridStackPPI",
    log_file: Optional[str] = None,
    **kwargs
) -> EnhancedLogger:
    """
    Factory function to create EnhancedLogger.
    
    Args:
        name: Logger name
        log_file: Optional log file path
        **kwargs: Additional arguments
    
    Returns:
        EnhancedLogger instance
    """
    return EnhancedLogger(name=name, log_file=log_file, **kwargs)
