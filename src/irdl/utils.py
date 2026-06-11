"""Utility functions for IRDL."""

from pathlib import Path
from types import ModuleType

import psutil

from irdl.logging import logger


def _get_dataset_classes(module: ModuleType) -> list[type]:
    """Return concrete BaseDataset subclasses exported by module."""
    from inspect import isabstract

    from irdl.base import BaseDataset

    dataset_classes: list[type] = []
    for name in dir(module):
        obj = getattr(module, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, BaseDataset)
            and not isabstract(obj)
            and hasattr(obj, "name")
            and hasattr(obj, "doi")
        ):
            dataset_classes.append(obj)
    return dataset_classes


def _fits_in_memory(ingest_path: Path) -> bool:
    """Check if a file can be loaded into available RAM.

    Needed when an entire dataset is loaded into memory.

    Parameters
    ----------
    ingest_path : Path
        Path to the ingestable file.

    Returns
    -------
    fits : bool
        True if the file fits into available RAM with headroom.
    """
    file_size = ingest_path.stat().st_size
    available = psutil.virtual_memory().available
    if file_size < available * 0.9:  # Headroom
        return True
    else:
        logger.warning(
            f"Dataset too large for available memory "
            f"({file_size / 1e9:.1f} GB needed, "
            f"{available / 1e9:.1f} GB available). "
        )
        return False
