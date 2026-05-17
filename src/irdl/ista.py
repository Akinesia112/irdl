"""Impulse response datasets from the Department of Engineering Acoustics, TU Berlin.

- MIRACLE: Microphone Array Impulse Response Dataset for Acoustic Learning.
- SRIRACHA: Shoebox Room Impulse Response Archive with Varying Absorption.

"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import h5py as h5
import numpy as np
import sofar as sf

from irdl.base import BaseDataset
from irdl.downloader import CACHE_DIR, _fetch, _pooch_from_doi


class IstaBaseDataset(BaseDataset):
    """Base class for HDF5-based datasets from ISTA (MIRACLE, SRIRACHA).

    Both MIRACLE and SRIRACHA share identical HDF5 file structure and can use
    the same ingestion logic to convert HDF5 to SOFA format.
    """

    # Room volume for SOFA metadata (subclasses must define)
    room_volume: float

    def ingest(self, file_path: Path) -> sf.Sofa:
        """Convert a MIRACLE/SRIRACHA HDF5 file into a SOFA object.

        Both datasets share an identical HDF5 layout, so this single
        implementation covers both subclasses. The output follows the
        SingleRoomMIMOSRIR SOFA convention.

        Parameters
        ----------
        file_path : Path
            Path to the HDF5 file.

        Returns
        -------
        sofar.Sofa
            SOFA object in the SingleRoomMIMOSRIR convention.
        """
        with h5.File(file_path, "r") as f:
            ir = f["data"]["impulse_response"][()]
            receiver_pos = f["data"]["location"]["receiver"][()]
            source_pos = f["data"]["location"]["source"][()]
            sampling_rate = f["metadata"]["sampling_rate"][()]
            temperature = f["metadata"]["temperature"][()]

        # SOFA dimension naming
        M, R, N = ir.shape  # number of measurements, receiver and samples
        E = 1  # number of emitters
        C = 3  # number of coordinates
        I = 1  # unity dimensions

        sofa = sf.Sofa("SingleRoomMIMOSRIR")

        # --- metadata  -------------------------------------------------
        sofa.GLOBAL_Title = self.name.upper()
        sofa.GLOBAL_AuthorContact = "a.pelling@tu-berlin.de; adam.kujawski@tu-berlin.de"
        sofa.GLOBAL_Organization = "TU Berlin, Department of Engineering Acoustics"
        sofa.GLOBAL_License = "CC BY-NC-SA 4.0"
        sofa.GLOBAL_References = self.doi
        sofa.GLOBAL_DatabaseName = self.name.upper()
        sofa.GLOBAL_RoomLocation = "TU Berlin, Einsteinufer 25"
        sofa.GLOBAL_ListenerShortName = "Custom planar microphone array"
        sofa.GLOBAL_ListenerDescription = (
            "64-channel planar microphone array "
            "(1.5 m × 1.5 m aluminium plate, Vogel's spiral, max spacing 1.47 m, 51.2 kHz sampling rate)"
        )
        sofa.GLOBAL_ReceiverShortName = "GRAS 40PL-1 Short CCP"
        sofa.GLOBAL_SourceShortName = "Loudspeaker"
        sofa.GLOBAL_SourceDescription = (
            "Dynamic 2\" cone loudspeaker in a cylindrical enclosure (Frequency range 100 Hz–16 kHz)"
        )

        sofa.RoomVolume = self.room_volume  # Dim. 1, M => so add a dimension upfront

        # --- environmental, per-measurement ------------------------------------
        sofa.RoomTemperature = temperature[..., np.newaxis]  # dim spec is (I, M)
        sofa.RoomTemperature_Units = "celsius"

        # --- geometry ----------------------------------------------------------
        # Receiver: fixed microphone array
        sofa.ReceiverPosition = receiver_pos.reshape(R, C, I)
        sofa.ReceiverPosition_Type = "cartesian"
        sofa.ReceiverPosition_Units = "metre"

        # Source: one cartesian position per measurement
        sofa.SourcePosition = source_pos  # dim spec is (M, C)

        # Emitter: single point source, co-located with the source frame origin
        sofa.EmitterPosition = np.zeros((E, C, I))
        sofa.EmitterPosition_Type = "cartesian"
        sofa.EmitterPosition_Units = "metre"

        # --- IR data -----------------------------------------------------------
        sofa.Data_IR = ir[..., np.newaxis]  # dim spec (M, R, N, E)
        sofa.Data_SamplingRate = np.full((I, M), sampling_rate)  # dim spec (I, M)
        sofa.Data_Delay = np.zeros((M, R, I))

        return sofa


class MiracleDataset(IstaBaseDataset):
    """MIRACLE: Microphone Array Impulse Response Dataset for Acoustic Learning."""

    name = "miracle"
    doi = "10.14279/depositonce-20837"
    room_volume = 830  # metadata needed for creation of sofa file

    def validate_params(self, **dataset_kwargs) -> None:
        """Validate MIRACLE-specific parameters.

        Parameters
        ----------
        **dataset_kwargs
            Dataset-specific parameters to validate. Expected keys: scenario, dataset_split.

        Raises
        ------
        ValueError
            If scenario or split is out of range, or 'D1' is combined with a split.
        """
        scenario = dataset_kwargs["scenario"]
        dataset_split = dataset_kwargs.get("dataset_split")

        if scenario not in ["A1", "A2", "D1", "R2"]:
            raise ValueError("scenario must be one of ['A1', 'A2', 'D1', 'R2']")
        if dataset_split not in [None, "C1", "C2", "C3", "C4"]:
            raise ValueError("dataset_split must be None or in [C1, C2, C3, C4]")
        if scenario == "D1" and dataset_split is not None:
            raise ValueError("scenario D1 cannot be split")

    @classmethod
    def get(
        cls,
        scenario: str = "A1",
        dataset_split: str | None = None,
        cache_dir: str | Path = CACHE_DIR,
        export_dir: str | Path | None = None,
        output_format: str = "pyfar",
    ):
        """Download MIRACLE dataset.

DOI: 10.14279/depositonce-20837

Parameters
----------
cache_dir : str
    Cache directory for downloads. Default: user cache directory.
export_dir : str, optional
    Directory for final output. Default: None (stays in cache_dir).
output_format : str
    Output format: 'pyfar', 'numpy', 'hdf5', 'sofa', or 'raw'.

scenario : str
    Scenario to download. One of 'A1', 'A2', 'D1', 'R2'.
dataset_split : str or None, optional
    Artificial dataset split. One of 'C1', 'C2', 'C3', 'C4' or None.
    Dense scenarios (D1) cannot be split.
"""
        instance = cls()
        return instance._get(
            scenario=scenario,
            dataset_split=dataset_split,
            cache_dir=cache_dir,
            export_dir=export_dir,
            output_format=output_format,
        )

    def _output_path(self, output_format: str, cache_dir: Path, export_dir: Path | None, **kwargs) -> Path | None:
        """Construct the output path for a MIRACLE file-based output.

        Parameters
        ----------
        output_format : str
            One of 'sofa', 'hdf5', 'raw'. Other formats return None.
        cache_dir : Path
            Cache directory.
        export_dir : Path or None
            Optional export directory; takes priority over cache_dir.
        **kwargs
            Must contain 'scenario'. May contain 'dataset_split'.

        Returns
        -------
        Path or None
            Canonical output path under '<base>/MIRACLE/', or None for in-memory formats.
        """
        if output_format not in ("sofa", "hdf5", "raw"):
            return None
        ext = ".sofa" if output_format == "sofa" else ".h5"
        scenario = kwargs["scenario"]
        split = kwargs.get("dataset_split")
        name = f"{scenario}{('-' + split) if split else ''}{ext}"
        base = (export_dir if export_dir is not None else cache_dir) / "MIRACLE"
        return base / name

    def _construct_file_name(self, **kwargs) -> str:
        """Construct HDF5 file name for MIRACLE (always full file, splits extracted later)."""
        scenario = kwargs["scenario"]
        # Always download the full file; splits are extracted in _get_file()
        return f"{scenario}.h5"

    def download(self, **kwargs) -> Path:
        """Download MIRACLE HDF5 file (always full scenario file).

        Parameters
        ----------
        **kwargs
            Dataset-specific parameters. Must contain 'cache_dir'.

        Returns
        -------
        Path
            Path to the downloaded file.
        """
        cache_dir = Path(kwargs["cache_dir"]) / "MIRACLE"
        cache_dir.mkdir(parents=True, exist_ok=True)

        file_name = self._construct_file_name(**kwargs)
        file_path = cache_dir / file_name

        # Download if not exists
        if not file_path.exists():
            pup = _pooch_from_doi(self.doi, path=cache_dir)
            _fetch(pup, file_name)

        return file_path

    def _get_file(self, cache_dir: Path, export_dir: Path | None, **kwargs) -> Path:
        """Check cache or download full file, then split if needed.

        Parameters
        ----------
        cache_dir : Path
            Base cache directory.
        export_dir : Path or None
            Optional export directory.
        **kwargs
            Dataset-specific parameters. Must contain 'scenario'. May contain 'dataset_split'.

        Returns
        -------
        Path
            Path to the file ready for ingest().
        """
        dataset_split = kwargs.get("dataset_split")

        # Download/get the full file using parent's _get_file
        file_path = super()._get_file(cache_dir, export_dir, **kwargs)

        # If a split is requested, extract it from the full file
        if dataset_split is not None:
            return self._extract_split(file_path, dataset_split, export_dir)

        return file_path

    def _extract_split(self, file_path: Path, dataset_split: str, export_dir: Path | None) -> Path:
        """Extract a dataset split from a full MIRACLE HDF5 file.

        Parameters
        ----------
        file_path : Path
            Path to the full HDF5 file.
        dataset_split : str
            Split to extract (C1, C2, C3, C4).
        export_dir : Path or None
            Optional directory for the extracted file.

        Returns
        -------
        Path
            Path to the extracted split HDF5 file.
        """
        # Load full data from HDF5
        with h5.File(file_path, "r") as f:
            data = {
                "impulse_response": f["data"]["impulse_response"][()],
                "receiver_coordinates": f["data"]["location"]["receiver"][()],
                "source_coordinates": f["data"]["location"]["source"][()],
                "speed_of_sound": f["metadata"]["c0"][()],
                "temperature": f["metadata"]["temperature"][()],
                "sampling_rate": f["metadata"]["sampling_rate"][()],
            }
            if "humidity" in f["metadata"]:
                data["humidity"] = f["metadata"]["humidity"][()]

        # Split to the requested quadrant
        offsets = {"C1": (0, 0), "C2": (0, 1), "C3": (1, 0), "C4": (1, 1)}
        row, column = offsets[dataset_split]
        n = int(np.sqrt(data["source_coordinates"].shape[0]))
        ir_shape = data["impulse_response"].shape
        data["source_coordinates"] = data["source_coordinates"].reshape(n, n, 3)[row::2, column::2, :].reshape(-1, 3)
        data["impulse_response"] = (
            data["impulse_response"].reshape(n, n, *ir_shape[1:])[row::2, column::2, :].reshape(-1, *ir_shape[1:])
        )

        # Save split data to a new HDF5 file
        split_file_name = f"{file_path.stem}-{dataset_split}{file_path.suffix}"
        if export_dir:
            split_path = Path(export_dir) / "MIRACLE" / split_file_name
        else:
            split_path = file_path.parent / split_file_name
        split_path.parent.mkdir(parents=True, exist_ok=True)

        with h5.File(split_path, "w") as f:
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

        return split_path


class SrirachaDataset(IstaBaseDataset):
    """SRIRACHA: Shoebox Room Impulse Response Archive with Varying Absorption."""

    name = "sriracha"
    doi = "10.14279/depositonce-23943"
    room_volume = 73.5

    def validate_params(self, **dataset_kwargs) -> None:
        """Validate SRIRACHA-specific parameters.

        Parameters
        ----------
        **dataset_kwargs
            Dataset-specific parameters to validate. Expected keys: scenario, dataset_split.

        Raises
        ------
        ValueError
            If scenario, split, or output_format combination is invalid.
        """
        output_format = dataset_kwargs.get("output_format")
        scenario = dataset_kwargs.get("scenario")
        dataset_split = dataset_kwargs.get("dataset_split")

        if scenario not in ["SR1", "SRA1", "SR1-D", "SRA1-D", "SR2", "SRA2", "SR2-D", "SRA2-D"]:
            raise ValueError("scenario must be one of [SR1, SRA1, SR1-D, SRA1-D, SR2, SRA2, SR2-D, SRA2-D]")
        if dataset_split not in [None, "C1", "C2", "C3", "C4"]:
            raise ValueError("dataset_split must be None or in [C1, C2, C3, C4]")
        if scenario.endswith("-D") and dataset_split is not None:
            raise ValueError("dense datasets do not have splits")
        if output_format == "raw" and scenario and not scenario.endswith("-D") and dataset_split is None:
            raise ValueError("raw output_format not supported for non-dense SRIRACHA scenarios without split")

    @classmethod
    def get(
        cls,
        scenario: str = "SR1-D",
        dataset_split: str | None = None,
        cache_dir: str | Path = CACHE_DIR,
        export_dir: str | Path | None = None,
        output_format: str = "pyfar",
    ):
        """Download SRIRACHA dataset.

DOI: 10.14279/depositonce-23943

Parameters
----------
cache_dir : str
    Cache directory for downloads. Default: user cache directory.
export_dir : str, optional
    Directory for final output. Default: None (stays in cache_dir).
output_format : str
    Output format: 'pyfar', 'numpy', 'hdf5', 'sofa', or 'raw'.

scenario : str
    Scenario to download. One of 'SR1', 'SRA1', 'SR1-D', 'SRA1-D', 'SR2',
    'SRA2', 'SR2-D', or 'SRA2-D'.
dataset_split : str or None, optional
    Optional dataset split for full-plane scenarios. One of 'C1', 'C2',
    'C3', 'C4', or None. Dense scenarios (ending in -D) do not have splits.
"""
        instance = cls()
        return instance._get(
            scenario=scenario,
            dataset_split=dataset_split,
            cache_dir=cache_dir,
            export_dir=export_dir,
            output_format=output_format,
        )

    def _output_path(self, output_format: str, cache_dir: Path, export_dir: Path | None, **kwargs) -> Path | None:
        """Construct the output path for a SRIRACHA file-based output.

        Parameters
        ----------
        output_format : str
            One of 'sofa', 'hdf5', 'raw'. Other formats return None.
        cache_dir : Path
            Cache directory.
        export_dir : Path or None
            Optional export directory; takes priority over cache_dir.
        **kwargs
            Must contain 'scenario'. May contain 'dataset_split'.

        Returns
        -------
        Path or None
            Canonical output path under '<base>/SRIRACHA/', or None for in-memory formats.
        """
        if output_format not in ("sofa", "hdf5", "raw"):
            return None
        ext = ".sofa" if output_format == "sofa" else ".h5"
        scenario = kwargs["scenario"]
        split = kwargs.get("dataset_split")
        name = f"{scenario}{('-' + split) if split else ''}{ext}"
        base = (export_dir if export_dir is not None else cache_dir) / "SRIRACHA"
        return base / name

    def _construct_file_name(self, **kwargs) -> str:
        """Construct HDF5 file name for SRIRACHA."""
        scenario = kwargs["scenario"]
        dataset_split = kwargs.get("dataset_split")
        if dataset_split:
            return f"{scenario}-{dataset_split}.h5"
        return f"{scenario}.h5"

    def download(self, **kwargs) -> Path:
        """Download SRIRACHA HDF5 file (single file only).

        Parameters
        ----------
        **kwargs
            Dataset-specific parameters. Must contain 'cache_dir'.

        Returns
        -------
        Path
            Path to the downloaded file.
        """
        cache_dir = Path(kwargs["cache_dir"]) / "SRIRACHA"
        cache_dir.mkdir(parents=True, exist_ok=True)

        file_name = self._construct_file_name(**kwargs)
        file_path = cache_dir / file_name

        # Download if not exists
        if not file_path.exists():
            pup = _pooch_from_doi(self.doi, path=cache_dir)
            _fetch(pup, file_name)

        return file_path

    def _get_file(self, cache_dir: Path, export_dir: Path | None, **kwargs) -> Path:
        """Check cache or download, with special handling for non-dense full scenarios."""
        scenario = kwargs["scenario"]
        dataset_split = kwargs.get("dataset_split")

        # For non-dense scenarios without split, need to download and merge 4 files
        if not scenario.endswith("-D") and dataset_split is None:
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

        return output_path
