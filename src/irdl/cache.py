"""Cache directory helpers for IRDL."""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterable
from pathlib import Path

import pooch as po

IRDL_CACHE_DIR = Path(os.getenv("IRDL_CACHE_DIR")) if "IRDL_CACHE_DIR" in os.environ else Path(po.os_cache("irdl"))

_CACHE_STAGES = ("provider", "ingest", "output")
_BINARY_UNITS = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
_BINARY_BASE = 1024.0


def resolve_cache_dir(cache_dir: Path | str | None = None) -> Path:
    """Return cache root for a custom path or IRDL default."""
    return IRDL_CACHE_DIR if cache_dir is None else Path(cache_dir)


def _path_size(path: Path) -> int:
    """Return total file size under path."""
    if path.is_file():
        return path.stat().st_size
    if not path.exists():
        return 0
    size = 0
    for child in path.rglob("*"):
        if child.is_file():
            size += child.stat().st_size
    return size


def _delete_path(path: Path) -> int:
    """Delete path and return bytes removed."""
    size = _path_size(path)
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)
    return size


def format_bytes(size: int, *, human_readable: bool = False) -> str:
    """Format byte count for CLI output."""
    if not human_readable:
        return str(size)

    value = float(size)
    for unit in _BINARY_UNITS:
        if value < _BINARY_BASE or unit == _BINARY_UNITS[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= _BINARY_BASE
    return f"{size} B"


def cache_size(cache_dir: Path | str | None = None) -> int:
    """Return total bytes stored under cache root."""
    return _path_size(resolve_cache_dir(cache_dir))


def cache_dir(cache_dir: Path | str | None = None) -> Path:
    """Return resolved cache root path."""
    return resolve_cache_dir(cache_dir)


def clean_cache(cache_dir: Path | str | None = None) -> int:
    """Remove cache root contents, recreate empty dir, and return freed bytes."""
    root = resolve_cache_dir(cache_dir)
    freed = _path_size(root)
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    return freed


def _stage_priority(path: Path) -> int:
    try:
        return _CACHE_STAGES.index(path.name)
    except ValueError:
        return -1


def _remove_children_except(root: Path, keep: Path) -> int:
    freed = 0
    for child in root.iterdir():
        if child == keep:
            continue
        freed += _delete_path(child)
    return freed


def prune_cache(
    cache_dir: Path | str | None = None,
    *,
    active_dataset_names: Iterable[str] | None = None,
) -> int:
    """Remove unreachable cache items and return freed bytes."""
    root = resolve_cache_dir(cache_dir)
    if not root.exists():
        return 0

    active_names = {name.lower() for name in active_dataset_names} if active_dataset_names is not None else None
    freed = 0

    for entry in root.iterdir():
        if entry.is_file() or entry.is_symlink():
            freed += _delete_path(entry)
            continue
        if not entry.is_dir():
            continue

        if active_names is not None and entry.name.lower() not in active_names:
            freed += _delete_path(entry)
            continue

        stages = [entry / stage for stage in _CACHE_STAGES if (entry / stage).exists()]
        if not stages:
            freed += _delete_path(entry)
            continue

        keep = max(stages, key=_stage_priority)
        freed += _remove_children_except(entry, keep)

    return freed
