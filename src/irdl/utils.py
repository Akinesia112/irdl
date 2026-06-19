"""Utility functions for IRDL."""

import os
import shutil
from pathlib import Path

from irdl.logging import logger


def _link_or_copy(source_path: Path, target_path: Path) -> Path:
    """Hard-link a file, falling back to copy."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.parent.parent == source_path.parent.parent:
        try:
            logger.debug(f"Linking {source_path} to {target_path}.")
            with logger.spin(f"Linking {target_path.name}..."):
                os.link(source_path, target_path)
        except OSError as e:
            logger.debug(f"Linking failed: {e!r}")
        else:
            return target_path
    logger.debug(f"Copying {source_path} to {target_path}.")
    with logger.spin(f"Copying {target_path.name}..."):
        shutil.copy2(source_path, target_path)
    return target_path
