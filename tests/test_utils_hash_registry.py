"""Tests for packaged hash-registry loading utilities."""

import json
from pathlib import Path

import pytest

from irdl.utils import _validate_hash_registry, load_hash_registry


def test_load_hash_registry_returns_cached_flat_mapping():
    """Verify packaged provider registries load as flat path->digest mappings."""
    registry_a = load_hash_registry("sofacoustics")
    registry_b = load_hash_registry("sofacoustics")

    assert registry_a is registry_b
    assert registry_a["hutubs/pp1_HRIRs_measured.sofa"].startswith("sha256:")


@pytest.mark.parametrize(
    ("payload", "error_type", "match"),
    [
        (["not", "an", "object"], TypeError, "must be a JSON object"),
        ({"hutubs": "sha256:abc"}, ValueError, "invalid path key"),
        ({"hutubs/file.sofa": "abc"}, ValueError, "invalid digest"),
    ],
)
def test_validate_hash_registry_rejects_invalid_shapes(payload, error_type, match):
    """Verify invalid packaged registry shapes fail fast."""
    with pytest.raises(error_type, match=match):
        _validate_hash_registry("sofacoustics", payload)


def test_sofacoustics_registry_json_exists_in_package_tree():
    """Verify the packaged registry file lives under the runtime package tree."""
    path = Path("src/irdl/registry/sofacoustics_hashes.json")

    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
