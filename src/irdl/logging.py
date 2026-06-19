"""Logging configuration for IRDL.

This module provides centralized logging setup for the irdl package using Rich.
"""

import io
import logging
import sys
from contextlib import contextmanager

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

# Create a Rich console for logging
console = Console()

# Single logger for the entire irdl module
logger = logging.getLogger("irdl")

# Configure Rich handler for the logger
rich_handler = RichHandler(
    console=console,
    show_time=False,
    show_path=False,
)
rich_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(rich_handler)
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.DEBUG)


class StdoutCapture:
    """Context manager to capture stdout and log it."""

    def __init__(self, logger_instance: logging.Logger = logger):
        self.logger = logger_instance

    def __enter__(self) -> io.StringIO:
        """Enter the context manager and start capturing stdout."""
        self.old_stdout = sys.stdout
        self.capture_buffer = io.StringIO()
        sys.stdout = self.capture_buffer
        return self.capture_buffer

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context manager and restore stdout."""
        sys.stdout = self.old_stdout
        output = self.capture_buffer.getvalue()
        if output:
            self.logger.info(output.strip())


# Create the context manager instance
as_stdout = StdoutCapture()

# Attach to logger for convenient access
logger.as_stdout = as_stdout


@contextmanager
def spin(message: str):
    """Show a Rich spinner while a blocking operation runs."""
    with console.status(message):
        yield


logger.spin = spin


def configure_cli_logging() -> logging.Logger:
    """Configure logging for CLI usage with Rich handler."""
    # Logger is already configured with RichHandler above
    # Just ensure it has the right level
    logger.setLevel(logging.INFO)
    return logger


# Make pooch use irdl's logger
try:
    import pooch as po

    pooch_logger = po.get_logger()

    # Remove pooch's existing handlers
    for handler in pooch_logger.handlers[:]:
        pooch_logger.removeHandler(handler)

    # Add a handler that forwards to irdl's logger
    class LoggerForwarder(logging.Handler):
        """Forward log records to a target logger."""

        def __init__(self, target_logger: logging.Logger) -> None:
            super().__init__()
            self.target_logger = target_logger

        def emit(self, record: logging.LogRecord) -> None:
            """Forward log records to the target logger."""
            # Re-emit the record with the target logger's name
            record.name = self.target_logger.name
            self.target_logger.handle(record)

    pooch_logger.addHandler(LoggerForwarder(logger))
    pooch_logger.propagate = False  # Don't propagate to root
    pooch_logger.setLevel(logging.DEBUG)
except ImportError:
    # pooch might not be available yet
    pass


class RichProgressBar:
    """Wrap rich.progress.Progress to satisfy the pooch progress bar interface.

    Pooch expects an object with a ``total`` attribute and ``update``, ``reset``, and
    ``close`` methods. This class provides that interface backed by a Rich progress bar.
    """

    def __init__(self, description: str, preset_total: int = 0):
        self._progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=console,
        )
        self._description = description
        self._task_id = None
        # Pooch sets self.total from the HTTP Content-Length header. If the server omits
        # that header, pooch sets it to 0. In that case, fall back to the preset value
        # from the repository API so the bar can show real progress.
        self._preset_total = preset_total
        self.total = 0

    @property
    def total(self) -> int:
        """Total download size in bytes.

        Returns
        -------
        int
            Total download size in bytes.
        """
        return self._total

    @total.setter
    def total(self, value: int) -> None:
        # Use the API-supplied size when the server omits Content-Length (value == 0).
        self._total = value or self._preset_total
        if self._task_id is not None:
            self._progress.update(self._task_id, total=self._total or None)

    def update(self, n: int) -> None:
        """Advance the progress bar by n bytes."""
        if self._task_id is None:
            self._progress.start()
            self._task_id = self._progress.add_task(self._description, total=self.total or None)
        self._progress.advance(self._task_id, n)

    def reset(self) -> None:
        """Reset the completed byte count to zero.

        Called by pooch before the final fill.
        """
        if self._task_id is not None:
            self._progress.reset(self._task_id, total=self.total or None)

    def close(self) -> None:
        """Fill to 100% and stop the progress display."""
        if self._task_id is not None:
            if self.total:
                self._progress.update(self._task_id, completed=self.total)
            self._progress.stop()
            self._task_id = None
