"""Small logging adapters used by Salvador command-line tools."""

from __future__ import annotations

import logging
from typing import Any, Protocol


class SupportsInfo(Protocol):
    """Protocol for logger-like objects used by :class:`Logger`."""

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None: ...


class Logger:
    """Thin wrapper around a console or file logger."""

    def __init__(self, logger: SupportsInfo) -> None:
        self.logger = logger

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log an informational message."""
        self.logger.info(msg, *args, **kwargs)


class ConsoleLogger:
    """Print informational messages to stdout when enabled."""

    def __init__(self, log_enabled: bool = True) -> None:
        self.log_enabled = log_enabled

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Print a formatted message when console logging is enabled."""
        if self.log_enabled:
            print(msg.format(*args))


class FileLogger:
    """Write informational messages to a log file."""

    def __init__(self, log_file: str = "app.log", log_level: int = logging.INFO) -> None:
        logging.basicConfig(
            filename=log_file,
            level=log_level,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a formatted informational message to the configured file."""
        self.logger.info(msg.format(*args))
