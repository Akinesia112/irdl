"""Tests for BaseDataset abstract base class."""

import pytest

from irdl.base import BaseDataset
from irdl.ista import IstaBaseDataset


class TestBaseDatasetAbstract:
    """Tests for BaseDataset abstract class behavior."""

    def test_cannot_instantiate_basedataset(self):
        """Verify BaseDataset cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseDataset()

    def test_abstract_methods_defined(self):
        """Verify BaseDataset has expected abstract methods."""
        expected_abstract = {
            "_source_filename",
            "download",
            "ingest",
            "validate_params",
        }
        assert BaseDataset.__abstractmethods__ == expected_abstract


class TestIstaBaseDatasetAbstract:
    """Tests for IstaBaseDataset abstract class behavior."""

    def test_cannot_instantiate_istabasedataset(self):
        """Verify IstaBaseDataset cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IstaBaseDataset()

    def test_inherits_abstract_methods(self):
        """Verify IstaBaseDataset inherits and adds abstract methods."""
        # IstaBaseDataset implements _source_filename and ingest, so only download and validate_params are abstract
        expected_abstract = {
            "download",
            "validate_params",
        }
        assert IstaBaseDataset.__abstractmethods__ == expected_abstract
