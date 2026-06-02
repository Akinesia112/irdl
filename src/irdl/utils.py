"""Utility functions for IRDL."""

from pathlib import Path

import psutil

from irdl.logging import logger


def _fits_in_memory(file_path: Path) -> bool:
    """Check if a file can be loaded into available RAM.

    Needed for pyfar or numpy output formats, which load the entire dataset into memory.

    Parameters
    ----------
    file_path : Path
        Path to the HDF5 file.

    Returns
    -------
    fits : bool
        True if the file fits into available RAM with headroom.
    """
    file_size = file_path.stat().st_size
    available = psutil.virtual_memory().available
    if file_size < available * 0.9:  # Headroom
        return True
    else:
        logger.warning(
            f"Dataset too large for available memory "
            f"({file_size / 1e9:.1f} GB needed, "
            f"{available / 1e9:.1f} GB available). "
            f"Returning HDF5 file path instead."
        )
        return False
