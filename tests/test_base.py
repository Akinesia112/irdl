"""Tests for BaseDataset abstract base class."""

from pathlib import Path

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
            "_download",
            "_ingest",
            "_validate_params",
        }
        assert BaseDataset.__abstractmethods__ == expected_abstract


class TestBaseDatasetHelpers:
    """Tests for shared BaseDataset helper behavior."""

    def test_output_path_accepts_source_filename_string(self, tmp_path):
        """Verify file-based output paths work when _source_filename returns a string."""

        class DummyDataset(BaseDataset):
            name = "dummy"
            doi = "10.0000/dummy"

            def _validate_params(self, **dataset_kwargs):
                pass

            def _download(self, provider_dir: Path, **_dataset_kwargs):
                return provider_dir / "dummy.sofa"

            def _ingest(self, ingest_path: Path):
                raise NotImplementedError

            def _source_filename(self, **_dataset_kwargs):
                return "dummy.sofa"

        dataset = DummyDataset()

        output_dir = tmp_path / "out"

        assert dataset._output_path(output_dir, "dummy.sofa", "sofa") == output_dir / "dummy.sofa"
        assert dataset._output_path(output_dir, "dummy.sofa", "hdf5") == output_dir / "dummy.h5"

    def test_process_passes_dataset_kwargs_to_subclass(self, tmp_path):
        """Verify the process wrapper forwards dataset-specific parameters to _process."""

        class DummyDataset(BaseDataset):
            name = "dummy"
            doi = "10.0000/dummy"

            def _validate_params(self, **dataset_kwargs):
                pass

            def _download(self, provider_dir: Path, **_dataset_kwargs):
                return provider_dir / "dummy.bin"

            def _ingest(self, ingest_path: Path):
                raise NotImplementedError

            def _source_filename(self, **_dataset_kwargs):
                return "dummy.bin"

            def _process(self, _provider_artifact: Path, ingest_path: Path, **dataset_kwargs):
                assert dataset_kwargs == {"scenario": "demo"}
                ingest_path.write_text("ok")
                return ingest_path

        dataset = DummyDataset()
        provider_artifact = tmp_path / "provider.bin"
        provider_artifact.write_text("provider")
        ingest_path = tmp_path / "ingest" / "dummy.bin"

        assert dataset.process(provider_artifact, ingest_path, scenario="demo") == ingest_path
        assert ingest_path.exists()


class TestIstaBaseDatasetAbstract:
    """Tests for IstaBaseDataset abstract class behavior."""

    def test_cannot_instantiate_istabasedataset(self):
        """Verify IstaBaseDataset cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IstaBaseDataset()

    def test_inherits_abstract_methods(self):
        """Verify IstaBaseDataset inherits and adds abstract methods."""
        # IstaBaseDataset implements _source_filename and _ingest,
        # so only _download and _validate_params are abstract
        expected_abstract = {
            "_download",
            "_validate_params",
        }
        assert IstaBaseDataset.__abstractmethods__ == expected_abstract
