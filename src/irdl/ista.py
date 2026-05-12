"""Impulse response datasets from the Department of Engineering Acoustics, TU Berlin.

- MIRACLE: Microphone Array Impulse Response Dataset for Acoustic Learning.
- SRIRACHA: Shoebox Room Impulse Response Archive with Varying Absorption.

"""

from pathlib import Path
from typing import Any

import h5py as h5
import numpy as np
import pooch as po
import pyfar as pf

from irdl.base import BaseDataset
from irdl.downloader import CACHE_DIR, _fetch, _pooch_from_doi
from irdl.utils import _move_to_export_dir

# =============================================================================
# Dataset Classes (Phase 2 Migration - ADR-0001)
# =============================================================================


class IstaBaseDataset(BaseDataset):
    """Base class for HDF5-based datasets from ISTA (MIRACLE, SRIRACHA).

    Both MIRACLE and SRIRACHA share identical HDF5 file structure and can use
    the same ingestion logic to convert HDF5 to SOFA format.
    """

    def ingest(self, file_path: Path) -> Any:
        """Convert HDF5 file to SOFA object.

        Shared implementation for MIRACLE and SRIRACHA since they use
        identical HDF5 structure.

        TODO: Implement HDF5 -> SOFA conversion
        """
        raise NotImplementedError("HDF5 to SOFA ingestion not yet implemented")


class MiracleDataset(IstaBaseDataset):
    """MIRACLE: Microphone Array Impulse Response Dataset for Acoustic Learning."""

    name = "miracle"
    doi = "10.14279/depositonce-20837"

    def validate_params(self, kwargs: dict) -> None:
        """Validate MIRACLE-specific parameters."""
        scenario = kwargs.get("scenario", "A1")
        dataset_split = kwargs.get("dataset_split")

        if scenario not in ["A1", "A2", "D1", "R2"]:
            raise ValueError("scenario must be one of ['A1', 'A2', 'D1', 'R2']")
        if dataset_split not in [None, "C1", "C2", "C3", "C4"]:
            raise ValueError("dataset_split must be None or in [C1, C2, C3, C4]")
        if scenario == "D1" and dataset_split is not None:
            raise ValueError("scenario D1 cannot be split")

    def _construct_file_name(self, **kwargs) -> str:
        """Construct HDF5 file name for MIRACLE."""
        scenario = kwargs.get("scenario", "A1")
        dataset_split = kwargs.get("dataset_split")
        if dataset_split:
            return f"{scenario}-{dataset_split}.h5"
        return f"{scenario}.h5"

    def download(self, **kwargs) -> Path:
        """Download MIRACLE HDF5 file."""
        cache_dir = Path(kwargs["cache_dir"]) / "MIRACLE"

        file_name = self._construct_file_name(**kwargs)
        file_path = cache_dir / file_name

        # Check if already exists (handled by _get_file, but download may be called directly)
        if file_path.exists():
            return file_path

        cache_dir.mkdir(parents=True, exist_ok=True)
        pup = _pooch_from_doi(self.doi, path=cache_dir)
        _fetch(pup, file_name)

        return file_path


class SrirachaDataset(IstaBaseDataset):
    """SRIRACHA: Shoebox Room Impulse Response Archive with Varying Absorption."""

    name = "sriracha"
    doi = "10.14279/depositonce-23943"

    def validate_params(self, kwargs: dict) -> None:
        """Validate SRIRACHA-specific parameters."""
        scenario = kwargs.get("scenario", "SR1-D")
        dataset_split = kwargs.get("dataset_split")
        output_format = kwargs.get("output_format", "pyfar")

        if scenario not in ["SR1", "SRA1", "SR1-D", "SRA1-D", "SR2", "SRA2", "SR2-D", "SRA2-D"]:
            raise ValueError("scenario must be one of [SR1, SRA1, SR1-D, SRA1-D, SR2, SRA2, SR2-D, SRA2-D]")
        if dataset_split not in [None, "C1", "C2", "C3", "C4"]:
            raise ValueError("dataset_split must be None or in [C1, C2, C3, C4]")
        if scenario[-1] == "D" and dataset_split is not None:
            raise ValueError("dense datasets do not have splits")
        # raw output_format not allowed for non-dense full scenarios
        if output_format == "raw" and scenario[-1] != "D" and dataset_split is None:
            raise ValueError("raw output_format not supported for non-dense SRIRACHA scenarios without split")

    def _construct_file_name(self, **kwargs) -> str:
        """Construct HDF5 file name for SRIRACHA."""
        scenario = kwargs.get("scenario", "SR1-D")
        dataset_split = kwargs.get("dataset_split")
        if dataset_split:
            return f"{scenario}-{dataset_split}.h5"
        return f"{scenario}.h5"

    def download(self, **kwargs) -> Path:
        """Download SRIRACHA HDF5 file (single file only)."""
        cache_dir = Path(kwargs["cache_dir"]) / "SRIRACHA"

        file_name = self._construct_file_name(**kwargs)
        file_path = cache_dir / file_name

        if file_path.exists():
            return file_path

        cache_dir.mkdir(parents=True, exist_ok=True)
        pup = _pooch_from_doi(self.doi, path=cache_dir)
        _fetch(pup, file_name)

        return file_path

    def _get_file(self, cache_dir: Path, export_dir: Path | None, **kwargs) -> Path:
        """Check cache or download, with special handling for non-dense full scenarios."""
        scenario = kwargs["scenario"]
        dataset_split = kwargs.get("dataset_split")

        # For non-dense scenarios without split, need to download and merge 4 files
        if scenario[-1] != "D" and dataset_split is None:
            return self._download_and_merge(scenario, cache_dir, export_dir, **kwargs)

        # For all other cases (dense, or with split), use normal flow
        return super()._get_file(cache_dir, export_dir, **kwargs)

    def _download_and_merge(self, scenario: str, cache_dir: Path, export_dir: Path | None, **kwargs) -> Path:
        """Download and merge four HDF5 split files into one full-plane dataset.

        This is the same logic as the module-level _download_and_merge but adapted
        for the Dataset class architecture.
        """
        path = cache_dir / "SRIRACHA"
        path.mkdir(parents=True, exist_ok=True)
        output_path = path / f"{scenario}.h5"

        offsets = {"C1": (0, 0), "C2": (0, 1), "C3": (1, 0), "C4": (1, 1)}

        # download split files
        split_files = {}
        pup = _pooch_from_doi(self.doi, path=path)
        for split in offsets:
            fname = f"{scenario}-{split}.h5"
            _fetch(pup, fname)
            split_files[split] = path / fname

        # read shapes and shared metadata from the first split
        with h5.File(split_files["C1"], "r") as f:
            ir_shape = f["data"]["impulse_response"].shape
            ir_dtype = f["data"]["impulse_response"].dtype
            n_split = ir_shape[0]
            sampling_rate = f["metadata"]["sampling_rate"][()]
            receiver = f["data"]["location"]["receiver"][()]
            has_humidity = "humidity" in f["metadata"]

        # calculate total number of sources and grid dimension
        n_sources = len(split_files) * n_split
        n_full_grid = int(np.sqrt(n_sources))
        n_split_grid = n_full_grid // 2

        with h5.File(output_path, "w") as out:
            # create groups and datasets
            data_grp = out.create_group("data")
            ir_ds = data_grp.create_dataset("impulse_response", shape=(n_sources, *ir_shape[1:]), dtype=ir_dtype)
            loc_grp = data_grp.create_group("location")
            src_ds = loc_grp.create_dataset("source", shape=(n_sources, 3), dtype="float64")
            loc_grp.create_dataset("receiver", data=receiver)

            meta_grp = out.create_group("metadata")
            meta_grp.create_dataset("sampling_rate", data=sampling_rate)
            c0_ds = meta_grp.create_dataset("c0", shape=(n_sources,), dtype="float32")
            temp_ds = meta_grp.create_dataset("temperature", shape=(n_sources,), dtype="float32")
            if has_humidity:
                hum_ds = meta_grp.create_dataset("humidity", shape=(n_sources,), dtype="float32")

            # open all split files
            handles = {s: h5.File(f, "r") for s, f in split_files.items()}
            try:
                # copy data from each split to the correct location in the output datasets
                for split_name, (row, col) in offsets.items():
                    f = handles[split_name]
                    for r in range(n_split_grid):
                        # index one row of the split grid
                        src = slice(r * n_split_grid, (r + 1) * n_split_grid)
                        # map split-grid-row to full-grid-row
                        grid_row = 2 * r + row
                        # index one row of the full grid, skipping every other entry
                        # to interleave splits
                        dst = slice(grid_row * n_full_grid + col, grid_row * n_full_grid + n_full_grid, 2)

                        ir_ds[dst] = f["data"]["impulse_response"][src]
                        src_ds[dst] = f["data"]["location"]["source"][src]
                        c0_ds[dst] = f["metadata"]["c0"][src]
                        temp_ds[dst] = f["metadata"]["temperature"][src]
                        if has_humidity:
                            hum_ds[dst] = f["metadata"]["humidity"][src]
            # close all files
            finally:
                for fh in handles.values():
                    fh.close()

            # delete split files
            for f in split_files.values():
                f.unlink()

        # Move to export_dir if specified
        if export_dir is not None:
            return _move_to_export_dir(output_path, export_dir)

        return output_path


# Create singleton instances
miracle_dataset = MiracleDataset()
sriracha_dataset = SrirachaDataset()


# =============================================================================
# Legacy Utility Functions (kept for backwards compatibility)
# =============================================================================


def _download_and_merge(scenario: str, path: Path, pup: po.Pooch):
    """Download and merge four HDF5 files into one full-plane dataset.

    Interleaves the datasplits into one dataset, writing
    row-by-row to keep memory usage bounded.

    Parameters
    ----------
    scenario : str
        Base scenario name, e.g. ``'SR1'``.
    path : Path
        Directory where HDF5 files are stored.
    pup : po.Pooch
        Pooch instance for downloading files.

    Returns
    -------
    output_path : Path
        Path to the merged HDF5 file.

    """
    output_path = path / f"{scenario}.h5"

    offsets = {"C1": (0, 0), "C2": (0, 1), "C3": (1, 0), "C4": (1, 1)}

    # download split files
    split_files = {}
    for split in offsets:
        fname = f"{scenario}-{split}.h5"
        _fetch(pup, fname)
        split_files[split] = path / fname

    # read shapes and shared metadata from the first split
    with h5.File(split_files["C1"], "r") as f:
        ir_shape = f["data"]["impulse_response"].shape
        ir_dtype = f["data"]["impulse_response"].dtype
        n_split = ir_shape[0]
        sampling_rate = f["metadata"]["sampling_rate"][()]
        receiver = f["data"]["location"]["receiver"][()]
        has_humidity = "humidity" in f["metadata"]

    # calculate total number of sources and grid dimension
    n_sources = len(split_files) * n_split
    n_full_grid = int(np.sqrt(n_sources))
    n_split_grid = n_full_grid // 2

    with h5.File(output_path, "w") as out:
        # create groups and datasets
        data_grp = out.create_group("data")
        ir_ds = data_grp.create_dataset("impulse_response", shape=(n_sources, *ir_shape[1:]), dtype=ir_dtype)
        loc_grp = data_grp.create_group("location")
        src_ds = loc_grp.create_dataset("source", shape=(n_sources, 3), dtype="float64")
        loc_grp.create_dataset("receiver", data=receiver)

        meta_grp = out.create_group("metadata")
        meta_grp.create_dataset("sampling_rate", data=sampling_rate)
        c0_ds = meta_grp.create_dataset("c0", shape=(n_sources,), dtype="float32")
        temp_ds = meta_grp.create_dataset("temperature", shape=(n_sources,), dtype="float32")
        if has_humidity:
            hum_ds = meta_grp.create_dataset("humidity", shape=(n_sources,), dtype="float32")

        # open all split files
        handles = {s: h5.File(f, "r") for s, f in split_files.items()}
        try:
            # copy data from each split to the correct location in the output datasets
            for split_name, (row, col) in offsets.items():
                f = handles[split_name]
                for r in range(n_split_grid):
                    # index one row of the split grid
                    src = slice(r * n_split_grid, (r + 1) * n_split_grid)
                    # map split-grid-row to full-grid-row
                    grid_row = 2 * r + row
                    # index one row of the full grid, skipping every other entry
                    # to interleave splits
                    dst = slice(grid_row * n_full_grid + col, grid_row * n_full_grid + n_full_grid, 2)

                    ir_ds[dst] = f["data"]["impulse_response"][src]
                    src_ds[dst] = f["data"]["location"]["source"][src]
                    c0_ds[dst] = f["metadata"]["c0"][src]
                    temp_ds[dst] = f["metadata"]["temperature"][src]
                    if has_humidity:
                        hum_ds[dst] = f["metadata"]["humidity"][src]
        # close all files
        finally:
            for fh in handles.values():
                fh.close()

        # delete split files
        for f in split_files.values():
            f.unlink()

    return output_path


def _load_h5(file: str):
    """Load raw arrays from an HDF5 file into a dictionary.

    Parameters
    ----------
    file : :class:`pathlib.Path` or :class:`str`
        Path to the HDF5 file.

    Returns
    -------
    data : :class:`dict`
        Dictionary with the following keys:

        - ``'impulse_response'`` : :class:`numpy.ndarray` — Impulse response data.
        - ``'receiver_coordinates'`` : :class:`numpy.ndarray` — Receiver positions as cartesian
          coordinates.
        - ``'source_coordinates'`` : :class:`numpy.ndarray` — Corrected source positions as
          cartesian coordinates.
        - ``'speed_of_sound'`` : :class:`numpy.ndarray` — Speed of sound per source position in
          m/s.
        - ``'temperature'`` : :class:`numpy.ndarray` — Ambient temperature per source position
          in °C.
        - ``'sampling_rate'`` : :class:`int` — Sampling rate in Hz.
        - ``'humidity'`` : :class:`numpy.ndarray` *(optional)* — Ambient humidity per source
          position, if present in the file.

    """
    with h5.File(file, "r") as f:
        data = {
            # data
            "impulse_response": f["data"]["impulse_response"][()],
            "receiver_coordinates": f["data"]["location"]["receiver"][()],
            "source_coordinates": f["data"]["location"]["source"][()],
            # metadata
            "speed_of_sound": f["metadata"]["c0"][()],
            "temperature": f["metadata"]["temperature"][()],
            "sampling_rate": f["metadata"]["sampling_rate"][()],
        }

        if "humidity" in f["metadata"]:
            data["humidity"] = f["metadata"]["humidity"][()]

    return data


def _split_data(data: dict, dataset_split: str):
    """Filter a data dictionary to a subgroup of source positions.

    Splits source positions and corresponding impulse responses into one of four
    dataset splits analogous to the ``dataset_split`` parameter in :func:`get_sriracha`.

    Parameters
    ----------
    data : :class:`dict`
        Dictionary of numpy arrays as returned by :func:`_load_h5`.
    dataset_split : :class:`str`
        Spatial quadrant to filter to. One of ``'C1'``, ``'C2'``, ``'C3'``, or ``'C4'``.

    Returns
    -------
    data : :class:`dict`
        Input dictionary with ``'source_coordinates'``, and ``'impulse_response'`` filtered
        to the requested dataset split.

    """
    # look up dictionary for the slicing indices
    offsets = {"C1": (0, 0), "C2": (0, 1), "C3": (1, 0), "C4": (1, 1)}
    row, column = offsets[dataset_split]

    # get array sizes for variable slicing
    n = int(np.sqrt(data["source_coordinates"].shape[0]))
    ir_shape = data["impulse_response"].shape

    # reshaping, slicing and reshape back to original shape
    data["source_coordinates"] = data["source_coordinates"].reshape(n, n, 3)[row::2, column::2, :].reshape(-1, 3)
    data["impulse_response"] = (
        data["impulse_response"].reshape(n, n, *ir_shape[1:])[row::2, column::2, :].reshape(-1, *ir_shape[1:])
    )

    return data


def _to_pyfar(data: dict):
    """Convert dictionary of arrays to pyfar objects.

    Converts impulse responses, source coordinates, and receiver coordinates to
    :class:`pyfar.Signal` and :class:`pyfar.Coordinates` respectively.

    Parameters
    ----------
    data : :class:`dict`
        Dictionary of numpy arrays as returned by :func:`_load_h5` or :func:`_split_data`.

    Returns
    -------
    data : :class:`dict`
        Dictionary with the following keys:

        - ``'impulse_response'`` : :class:`pyfar.Signal` — Impulse response data.
        - ``'source_coordinates'`` : :class:`pyfar.Coordinates` — Corrected source positions.
        - ``'receiver_coordinates'`` : :class:`pyfar.Coordinates` — Receiver positions.

    """
    data["impulse_response"] = pf.Signal(data["impulse_response"], sampling_rate=data["sampling_rate"])
    data["source_coordinates"] = pf.Coordinates(*data["source_coordinates"].T)
    data["receiver_coordinates"] = pf.Coordinates(*data["receiver_coordinates"].T)

    for key in ["sampling_rate", "speed_of_sound", "temperature", "humidity"]:
        data.pop(key, None)

    return data


def _save_h5(data: dict, path: str):
    """Save a data dictionary of numpy arrays to an HDF5 file.

    Helper function for artificially split data. Writes the contents of a data
    dictionary as returned by :func:`_load_h5` or :func:`_split_data` to an HDF5
    file following the same structure as the MIRACLE and SRIRACHA datasets.

    Parameters
    ----------
    data : :class:`dict`
        Dictionary of numpy arrays as returned by :func:`_load_h5` or :func:`_split_data`.
    path : :class:`pathlib.Path` or :class:`str`
        Path to the HDF5 file to write.

    Returns
    -------
    path : :class:`pathlib.Path`
        Path to the written HDF5 file.

    """
    with h5.File(path, "w") as f:
        data_group = f.create_group("data")
        data_group.create_dataset("impulse_response", data=data["impulse_response"])
        location_group = data_group.create_group("location")
        location_group.create_dataset("source", data=data["source_coordinates"])
        location_group.create_dataset("receiver", data=data["receiver_coordinates"])
        metadata_group = f.create_group("metadata")
        metadata_group.create_dataset("c0", data=data["speed_of_sound"])
        metadata_group.create_dataset("temperature", data=data["temperature"])
        metadata_group.create_dataset("sampling_rate", data=data["sampling_rate"])
        if "humidity" in data:
            metadata_group.create_dataset("humidity", data=data["humidity"])
    return path


def get_miracle(
    scenario: str = "A1",
    dataset_split: str = None,
    cache_dir: str = CACHE_DIR,
    export_dir: str = None,
    output_format: str = "pyfar",
):
    """Download and extract the MIRACLE database from DepositOnce.

    DOI: `10.14279/depositonce-20837 <https://doi.org/10.14279/depositonce-20837>`_

    Parameters
    ----------
    scenario : :class:`str`
        Name of the scenario to download. Either ``'A1'``, ``'A2'``, ``'D1'`` or ``'R2'``.
    dataset_split : :class:`str` or None
        Artificial dataset split. Analogous to ``dataset_split`` in :func:`get_sriracha`.
        One of ``'C1'``, ``'C2'``, ``'C3'``, ``'C4'``, or ``None`` (default).
    cache_dir : :class:`str` or :class:`pathlib.Path`
        Directory used to store raw downloads and intermediate files. Overridden
        by the environment variable ``IRDL_CACHE_DIR`` when set. Defaults to the
        user cache directory.
    export_dir : :class:`str` or :class:`pathlib.Path` or None
        Directory to move the output file to after processing. When ``None``
        (default) the output file stays in ``cache_dir``.
    output_format : :class:`str`
        Output format of the returned data.
        Either ``'pyfar'`` (default), ``'hdf5'``, ``'numpy'``, or ``'raw'``.

    Returns
    -------
    data : :class:`dict` or :class:`pathlib.Path`
        Returned data depends on ``output_format``:

        - ``'pyfar'``: :class:`dict` with keys ``'impulse_response'`` (:class:`pyfar.Signal`),
          ``'source_coordinates'`` (:class:`pyfar.Coordinates`),
          ``'receiver_coordinates'`` (:class:`pyfar.Coordinates`)
        - ``'hdf5'``: :class:`pathlib.Path` to the HDF5 file containing the data.
        - ``'numpy'``: :class:`dict` with keys ``'impulse_response'`` (:class:`numpy.ndarray`),
          ``'source_coordinates'`` (:class:`numpy.ndarray`),
          ``'receiver_coordinates'`` (:class:`numpy.ndarray`),
          ``'speed_of_sound'`` (:class:`numpy.ndarray`),
          ``'temperature'`` (:class:`numpy.ndarray`),
          ``'sampling_rate'`` (:class:`int`)
        - ``'raw'``: :class:`pathlib.Path` to the raw HDF5 file.

    """
    return miracle_dataset.get(
        scenario=scenario,
        dataset_split=dataset_split,
        cache_dir=cache_dir,
        export_dir=export_dir,
        output_format=output_format,
    )


def get_sriracha(
    scenario: str = "SR1-D",
    dataset_split: str = None,
    cache_dir: str = CACHE_DIR,
    export_dir: str = None,
    output_format: str = "pyfar",
):
    """Download and extract the SRIRACHA database from DepositOnce.

    DOI: `10.14279/depositonce-23943 <https://doi.org/10.14279/depositonce-23943>`_

    Parameters
    ----------
    scenario : :class:`str`
        Name of the scenario to download. One of ``'SR1'``, ``'SRA1'``, ``'SR1-D'``,
        ``'SRA1-D'``, ``'SR2'``, ``'SRA2'``, ``'SR2-D'``, or ``'SRA2-D'``.
    dataset_split : :class:`str` or None
        Optional dataset split for full-plane scenarios.
        One of ``'C1'``, ``'C2'``, ``'C3'``, ``'C4'``, or ``None`` (default).
        Dense scenarios (ending in ``-D``) do not have splits.
    cache_dir : :class:`str` or :class:`pathlib.Path`
        Directory used to store raw downloads and intermediate files. Overridden
        by the environment variable ``IRDL_CACHE_DIR`` when set. Defaults to the
        user cache directory.
    export_dir : :class:`str` or :class:`pathlib.Path` or None
        Directory to move the output file to after processing. When ``None``
        (default) the output file stays in ``cache_dir``.
    output_format : :class:`str`
        Output format of the returned data.
        Either ``'pyfar'`` (default), ``'hdf5'``, ``'numpy'``, or ``'raw'``.

    Returns
    -------
    data : :class:`dict` or :class:`pathlib.Path`
        Returned data depends on ``output_format``:

        - ``'pyfar'``: :class:`dict` with keys ``'impulse_response'`` (:class:`pyfar.Signal`),
          ``'source_coordinates'`` (:class:`pyfar.Coordinates`), and
          ``'receiver_coordinates'`` (:class:`pyfar.Coordinates`).
        - ``'hdf5'``: :class:`pathlib.Path` to the HDF5 file containing the data.
        - ``'numpy'``: :class:`dict` with keys ``'impulse_response'`` (:class:`numpy.ndarray`),
          ``'source_coordinates'`` (:class:`numpy.ndarray`),
          ``'receiver_coordinates'`` (:class:`numpy.ndarray`),
          ``'speed_of_sound'`` (:class:`numpy.ndarray`),
          ``'temperature'`` (:class:`numpy.ndarray`),
          ``'sampling_rate'`` (:class:`int`), and optionally
          ``'humidity'`` (:class:`numpy.ndarray`).
        - ``'raw'``: :class:`pathlib.Path` to the raw HDF5 file.

    """
    return sriracha_dataset.get(
        scenario=scenario,
        dataset_split=dataset_split,
        cache_dir=cache_dir,
        export_dir=export_dir,
        output_format=output_format,
    )
