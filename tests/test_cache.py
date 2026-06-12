"""Tests for cache helpers."""

from irdl import cache as cache_module
from irdl.cache import cache_size, clean_cache, format_bytes, prune_cache, resolve_cache_dir


def test_format_bytes_human_readable():
    """Verify byte formatting uses binary units."""
    assert format_bytes(1536, human_readable=True) == "1.5 KiB"
    assert format_bytes(1536, human_readable=False) == "1536"


def test_resolve_cache_dir_uses_default_path(tmp_path, monkeypatch):
    """Verify default cache dir respects IRDL_CACHE_DIR."""
    monkeypatch.setattr(cache_module, "IRDL_CACHE_DIR", tmp_path / "cache")

    assert resolve_cache_dir() == tmp_path / "cache"


def test_clean_cache_recreates_empty_root(tmp_path):
    """Verify clean removes contents and recreates cache root."""
    root = tmp_path / "cache"
    file_path = root / "dataset" / "file.bin"
    file_path.parent.mkdir(parents=True)
    file_path.write_bytes(b"abc")

    freed = clean_cache(root)

    assert freed == len(b"abc")
    assert root.exists()
    assert list(root.iterdir()) == []


def test_cache_size_counts_all_files(tmp_path):
    """Verify cache size includes every file under root."""
    root = tmp_path / "cache"
    (root / "dataset" / "provider").mkdir(parents=True)
    (root / "dataset" / "provider" / "a.bin").write_bytes(b"abcd")
    (root / "dataset" / "output").mkdir(parents=True)
    (root / "dataset" / "output" / "b.bin").write_bytes(b"ef")

    assert cache_size(root) == len(b"abcd") + len(b"ef")


def test_prune_keeps_highest_stage_only(tmp_path):
    """Verify prune keeps only highest reachable stage."""
    root = tmp_path / "cache"
    dataset = root / "fabian"
    (dataset / "provider").mkdir(parents=True)
    (dataset / "provider" / "provider.bin").write_bytes(b"provider")
    (dataset / "ingest").mkdir()
    (dataset / "ingest" / "ingest.sofa").write_bytes(b"ingest")
    (dataset / "output").mkdir()
    (dataset / "output" / "output.sofa").write_bytes(b"output")

    old_dataset = root / "old_dataset"
    (old_dataset / "provider").mkdir(parents=True)
    (old_dataset / "provider" / "old.bin").write_bytes(b"old")

    freed = prune_cache(root, active_dataset_names={"fabian"})

    assert freed == len(b"provider") + len(b"ingest") + len(b"old")
    assert (dataset / "output").exists()
    assert not (dataset / "provider").exists()
    assert not (dataset / "ingest").exists()
    assert not old_dataset.exists()
