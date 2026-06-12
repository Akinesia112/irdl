"""Tests for dataset category attributes on dataset classes."""

import irdl
from irdl.base import DatasetCategory, _get_dataset_classes


def _public_dataset_classes() -> list[type]:
    return _get_dataset_classes(irdl)


class TestDatasetCategories:
    """Tests for dataset category attributes."""

    def test_every_public_dataset_has_category(self):
        """Verify all public Datasets have a _category attribute."""
        for cls in _public_dataset_classes():
            assert hasattr(cls, "_category"), f"{cls.__name__} is missing _category attribute"
            assert isinstance(cls._category, DatasetCategory), (
                f"{cls.__name__}._category is not a DatasetCategory"
            )

    def test_all_categories_are_known(self):
        """Verify all dataset categories are from the known DatasetCategory enum."""
        for cls in _public_dataset_classes():
            category = getattr(cls, "_category", None)
            assert category is not None, f"{cls.__name__} has no category"
            # Check it's a valid enum value
            assert category in DatasetCategory, f"{cls.__name__} has unknown category: {category}"

    def test_category_distribution(self):
        """Verify datasets are distributed across expected categories."""
        from collections import Counter

        categories = [cls._category for cls in _public_dataset_classes()]
        category_counts = Counter(categories)

        # At least one dataset per category that exists
        for category, count in category_counts.items():
            assert count > 0, f"Category {category} has no datasets"

        # Verify we have the expected categories
        assert DatasetCategory.ROOM_IMPULSE_RESPONSES in category_counts, (
            "Expected ROOM_IMPULSE_RESPONSES category to be used"
        )
        assert DatasetCategory.HEAD_RELATED_IMPULSE_RESPONSES in category_counts, (
            "Expected HEAD_RELATED_IMPULSE_RESPONSES category to be used"
        )
