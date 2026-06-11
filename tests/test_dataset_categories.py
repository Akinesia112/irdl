"""Tests for docs/dataset_categories.py."""

from importlib import util
from pathlib import Path

import irdl

from irdl.utils import _get_dataset_classes


def _load_dataset_categories():
    path = Path(__file__).resolve().parents[1] / "docs" / "dataset_categories.py"
    spec = util.spec_from_file_location("dataset_categories", path)
    module = util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module.DATASET_CATEGORIES


def _public_dataset_names() -> set[str]:
    return {f"irdl.{cls.__name__}" for cls in _get_dataset_classes(irdl)}


class TestDatasetCategories:
    """Tests for the dataset category single source of truth."""

    def test_every_public_dataset_has_category_entry(self):
        """Verify docs dataset mapping covers all public Datasets."""
        categories = _load_dataset_categories()
        category_names = {
            dataset["name"]
            for category in categories
            for dataset in category["datasets"]
        }

        assert category_names == _public_dataset_names()
