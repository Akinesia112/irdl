"""Tests for ISTA-specific file processing helpers."""

from pathlib import Path

import h5py
import numpy as np

from irdl.ista import MiracleDataset, SrirachaDataset


def _write_ista_hdf5(path: Path, *, n_sources: int, start: int = 0) -> None:
    """Write a minimal ISTA-style HDF5 file for processing tests."""
    impulse_response = np.arange(start, start + n_sources * 2 * 8, dtype=np.float32).reshape(n_sources, 2, 8)
    source_coordinates = np.arange(start, start + n_sources * 3, dtype=np.float64).reshape(n_sources, 3)
    receiver_coordinates = np.array([[0.0, 0.5, 0.0], [0.0, -0.5, 0.0]], dtype=np.float64)
    c0 = np.full(n_sources, 343.0, dtype=np.float32)
    temperature = np.full(n_sources, 20.0, dtype=np.float32)

    with h5py.File(path, "w") as handle:
        data_group = handle.create_group("data")
        data_group.create_dataset("impulse_response", data=impulse_response)
        location_group = data_group.create_group("location")
        location_group.create_dataset("source", data=source_coordinates)
        location_group.create_dataset("receiver", data=receiver_coordinates)

        metadata_group = handle.create_group("metadata")
        metadata_group.create_dataset("sampling_rate", data=44100)
        metadata_group.create_dataset("c0", data=c0)
        metadata_group.create_dataset("temperature", data=temperature)


class TestMiracleProcessing:
    """Tests for MIRACLE-specific processing helpers."""

    def test_extract_split_preserves_permissions(self, tmp_path):
        """Verify extracted split files reuse the source file's permission bits."""
        dataset = MiracleDataset()
        provider_artifact = tmp_path / "A1.h5"
        _write_ista_hdf5(provider_artifact, n_sources=4)
        provider_artifact.chmod(0o640)

        output_path = tmp_path / "ingest" / "A1-C1.h5"
        result = dataset._extract_split(provider_artifact, "C1", output_path)

        assert result.stat().st_mode & 0o777 == provider_artifact.stat().st_mode & 0o777


class TestSrirachaProcessing:
    """Tests for SRIRACHA-specific processing helpers."""

    def test_merge_split_files_preserves_permissions(self, tmp_path):
        """Verify merged ingest files reuse the split file's permission bits."""
        dataset = SrirachaDataset()
        provider_dir = tmp_path / "provider"
        provider_dir.mkdir()

        for index, split in enumerate(("C1", "C2", "C3", "C4")):
            split_path = provider_dir / f"SR1-{split}.h5"
            _write_ista_hdf5(split_path, n_sources=1, start=index * 100)
            split_path.chmod(0o640)

        ingest_path = tmp_path / "ingest" / "SR1.h5"
        ingest_path.parent.mkdir()
        result = dataset._merge_split_files("SR1", provider_dir, ingest_path)

        assert result.stat().st_mode & 0o777 == 0o640  # noqa: PLR2004
        assert not any(provider_dir.glob("SR1-C*.h5"))
