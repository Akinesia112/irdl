"""Impulse response datasets from the Department of Engineering Acoustics, TU Berlin.

- MIRACLE: Microphone Array Impulse Response Dataset for Acoustic Learning.
- SRIRACHA: Shoebox Room Impulse Response Archive with Varying Absorption.

"""

from pathlib import Path

import h5py as h5
import numpy as np
import sofar as sf

from irdl.base import BaseDataset
from irdl.downloader import _fetch, _pooch_from_doi
from irdl.logging import logger


class IstaBaseDataset(BaseDataset):
    """Base class for HDF5-based datasets from ISTA (MIRACLE, SRIRACHA).

    Both MIRACLE and SRIRACHA share identical HDF5 file structure and can use
    the same ingestion logic to convert HDF5 to SOFA format.

    Attributes
    ----------
    room_volume : float
        Room volume in cubic meters, used for SOFA metadata.
    """

    def _source_filename(self, **dataset_kwargs) -> str:
        """Construct the raw input filename with extension.

        Shared implementation for MIRACLE and SRIRACHA datasets.

        Parameters
        ----------
        **dataset_kwargs : dict
            Must contain 'scenario'. May contain 'dataset_split'.

        Returns
        -------
        str
            Filename in format "{scenario}[-{split}].h5".
        """
        scenario = dataset_kwargs["scenario"]
        split = dataset_kwargs.get("dataset_split")
        return f"{scenario}{('-' + split) if split else ''}.h5"

    def _ingest(self, ingest_path: Path) -> sf.Sofa:
        """Convert a MIRACLE/SRIRACHA HDF5 file into a SOFA object.

        Both datasets share an identical HDF5 layout, so this single
        implementation covers both subclasses. The output follows the
        SingleRoomMIMOSRIR SOFA convention.

        Parameters
        ----------
        ingest_path : :class:`pathlib.Path`
            Path to the HDF5 file.

        Returns
        -------
        :class:`sofar.Sofa`
            SOFA object in the SingleRoomMIMOSRIR convention.
        """
        with h5.File(ingest_path, "r") as f:
            ir = f["data"]["impulse_response"][()]
            receiver_pos = f["data"]["location"]["receiver"][()]
            source_pos = f["data"]["location"]["source"][()]
            sampling_rate = f["metadata"]["sampling_rate"][()]
            temperature = f["metadata"]["temperature"][()]
            speed_of_sound = f["metadata"]["c0"][()]
            humidity = f["metadata"]["humidity"][()] if "humidity" in f["metadata"] else None

        # SOFA dimension naming
        m, r, n = ir.shape  # number of measurements, receiver and samples
        e = 1  # number of emitters
        c = 3  # number of coordinates
        i = 1  # unity dimensions

        sofa = sf.Sofa("SingleRoomMIMOSRIR")

        # --- metadata  -------------------------------------------------
        sofa.GLOBAL_Title = self.name.upper()
        sofa.GLOBAL_AuthorContact = "a.pelling@tu-berlin.de; adam.kujawksi@tu-berlin.de"
        sofa.GLOBAL_Organization = "TU Berlin, Department of Engineering Acoustics"
        sofa.GLOBAL_License = "CC BY-NC-SA 4.0"
        sofa.GLOBAL_References = self.doi
        sofa.GLOBAL_DatabaseName = self.name.upper()
        sofa.GLOBAL_RoomLocation = "TU Berlin, Einsteinufer 25, 10587 Berlin"
        sofa.GLOBAL_ListenerShortName = "Custom planar microphone array"
        sofa.GLOBAL_ListenerDescription = (
            "64-channel planar microphone array "
            "(1.5 m × 1.5 m aluminium plate, Vogel's spiral, max spacing 1.47 m, 51.2 kHz sampling rate)"
        )
        sofa.GLOBAL_ReceiverShortName = "GRAS 40PL-1 Short CCP"
        sofa.GLOBAL_SourceShortName = "Loudspeaker"
        sofa.GLOBAL_SourceDescription = (
            "Dynamic 2” cone loudspeaker in a cylindrical enclosure (Frequency range 100 Hz–16 kHz)"
        )

        sofa.RoomVolume = self.room_volume  # #Dim. 1, M => so add a dimension upfront
        sofa.MeasurementDate = np.full(m, self.measurement_date)  # (M,)

        # --- environmental  ----------------------------------------------------
        sofa.RoomTemperature = temperature[np.newaxis, ...] + 273.15  # C to K
        sofa.RoomTemperature_Units = "kelvin"

        # --- geometry ----------------------------------------------------------
        # Listener: whole array
        sofa.ListenerPosition = np.zeros((m, c))  # fixed array origin, one row per measurement (M, C)
        sofa.ListenerPosition_Type = "cartesian"
        sofa.ListenerPosition_Units = "metre"

        # Receiver: microphones
        sofa.ReceiverPosition = receiver_pos.reshape(r, c, i)
        sofa.ReceiverPosition_Type = "cartesian"
        sofa.ReceiverPosition_Units = "metre"
        sofa.ReceiverDescriptions = np.array(["GRAS 40PL-1 Short CCP"] * r)  # (R, S)
        sofa.ReceiverView = np.tile([1.0, 0.0, 0.0], (r, 1))[..., np.newaxis]  # look +x, (R, C, I)
        sofa.ReceiverUp = np.tile([0.0, 0.0, 1.0], (r, 1))[..., np.newaxis]  # up +z,  (R, C, I)

        # Source: source frame; one cartesian position per measurement
        sofa.SourcePosition = source_pos  ##dim spec is (M, C)

        # Emitter: single point source, co-located with the source frame origin
        sofa.EmitterPosition = np.zeros((e, c, i))
        sofa.EmitterPosition_Type = "cartesian"
        sofa.EmitterPosition_Units = "metre"

        # --- IR data -----------------------------------------------------------
        sofa.Data_IR = ir[..., np.newaxis]  # dim spec (M, R, N, E)
        sofa.Data_SamplingRate = np.full((i, m), sampling_rate)  # dim spec (I, M)
        sofa.Data_Delay = np.zeros((m, r, i))

        # --- Custom data (not part of the SOFA convention) ---------------------
        sofa.add_variable("SpeedOfSound", speed_of_sound.reshape(m, i), "double", "MI")
        if humidity is not None:
            sofa.add_variable("Humidity", humidity.reshape(m, i), "double", "MI")

        return sofa


class MiracleDataset(IstaBaseDataset):
    """Download and extract the MIRACLE database from DepositOnce.

    Attributes
    ----------
    name : str
        Dataset name ("miracle").
    doi : str
        Digital Object Identifier ("10.14279/depositonce-20837").
    room_volume : float
        Room volume in cubic meters (830).
    measurement_date : float
        Release date in POSIX seconds, used as the SOFA MeasurementDate
        (no per-measurement date is available).
    """

    name = "miracle"
    doi = "10.14279/depositonce-20837"
    # metadata needed for creation of sofa file
    room_volume = 830
    measurement_date = 1697068800.0

    @classmethod
    def get(
        cls,
        scenario: str = "A1",
        dataset_split: str | None = None,
        cache_dir: str | Path | None = None,
        export_dir: str | Path | None = None,
        output_format: str = "pyfar",
    ) -> dict | Path | None:
        """
        scenario : str
            Scenario to download. One of 'A1', 'A2', 'D1', 'R2'.
        dataset_split : str or None, optional
            Artificial dataset split. One of 'C1', 'C2', 'C3', 'C4' or None.
            Dense scenarios (D1) cannot be split.

        Returns
        -------
        dict or Path
            For 'pyfar' / 'numpy': dict of in-memory objects.
            For 'sofa' / 'hdf5' / 'raw': Path to file on disk.
        """  # noqa: D205, D403
        return cls()._get(
            scenario=scenario,
            dataset_split=dataset_split,
            cache_dir=cache_dir,
            export_dir=export_dir,
            output_format=output_format,
        )

    def _validate_params(self, **dataset_kwargs) -> None:
        """Validate MIRACLE-specific parameters.

        Parameters
        ----------
        **dataset_kwargs : dict
            Must contain 'scenario' (one of 'A1', 'A2', 'D1', 'R2'). May
            contain 'dataset_split' (one of 'C1', 'C2', 'C3', 'C4', or None).
            Scenario 'D1' cannot be split. ``output_format`` is also passed
            but unused here.

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
            raise ValueError("dataset_split must be None or one ['C1', 'C2', 'C3', 'C4']")
        if scenario == "D1" and dataset_split is not None:
            raise ValueError("scenario D1 cannot be split")

    def _download(self, provider_dir: Path, **dataset_kwargs) -> Path:
        """Download MIRACLE dataset file.

        Downloads the full scenario HDF5 file. If a split is requested,
        the split will be extracted in _process().

        Parameters
        ----------
        provider_dir : :class:`pathlib.Path`
            Provider directory (e.g., ``cache/MIRACLE/provider/``).
        **dataset_kwargs : dict
            Must contain 'scenario'. May contain 'dataset_split'.

        Returns
        -------
        :class:`pathlib.Path`
            Path to the downloaded full scenario HDF5 file inside the
            provider directory.
        """
        full_path = provider_dir / self._source_filename(**{**dataset_kwargs, "dataset_split": None})
        logger.info(f"Downloading MIRACLE scenario {dataset_kwargs['scenario']}")
        pup = _pooch_from_doi(self.doi, path=provider_dir)
        _fetch(pup, full_path.name)
        return full_path

    def _process(self, provider_artifact: Path, ingest_path: Path, **dataset_kwargs) -> Path:
        """Post-process MIRACLE file if needed.

        If a dataset_split is requested and the file is the full scenario file,
        extracts the corresponding quadrant split into the ingest directory.
        Otherwise promotes the provider file to the ingest stage.

        Parameters
        ----------
        provider_artifact : :class:`pathlib.Path`
            Path to the provider file (full scenario HDF5).
        ingest_path : :class:`pathlib.Path`
            Path to the HDF5 file in the ingest directory.
        **dataset_kwargs : dict
            Must contain 'scenario'. May contain 'dataset_split'.

        Returns
        -------
        :class:`pathlib.Path`
            Path to the processed file in the ingest directory.
        """
        split = dataset_kwargs.get("dataset_split")

        # If no split requested, promote to ingest stage
        if not split:
            return super()._process(provider_artifact, ingest_path, **dataset_kwargs)
        return self._extract_split(provider_artifact, split, ingest_path)

    def _extract_split(self, ingest_path: Path, dataset_split: str, output_path: Path) -> Path:
        """Extract a dataset split from a full MIRACLE HDF5 file.

        Reads the full file, indexes the requested quadrant of the source
        grid, and writes the result to a new HDF5 file.

        Parameters
        ----------
        ingest_path : :class:`pathlib.Path`
            Path to the full HDF5 file in the provider directory.
        dataset_split : str
            Split to extract. One of 'C1', 'C2', 'C3', 'C4'.
        output_path : :class:`pathlib.Path`
            Target path in the ingest directory.

        Returns
        -------
        :class:`pathlib.Path`
            Path to the extracted split HDF5 file.
        """
        logger.info(f"Extracting split {dataset_split} from {ingest_path.name}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Load full data from HDF5
        with h5.File(ingest_path, "r") as f:
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
        data["temperature"] = data["temperature"].reshape(n, n)[row::2, column::2].reshape(-1)
        data["speed_of_sound"] = data["speed_of_sound"].reshape(n, n)[row::2, column::2].reshape(-1)
        if "humidity" in data:
            data["humidity"] = data["humidity"].reshape(n, n)[row::2, column::2].reshape(-1)

        with h5.File(output_path, "w") as f:
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

        return output_path


class SrirachaDataset(IstaBaseDataset):
    """Download and extract the SRIRACHA database from DepositOnce.

    Attributes
    ----------
    name : str
        Dataset name ("sriracha").
    doi : str
        Digital Object Identifier ("10.14279/depositonce-23943").
    room_volume : float
        Room volume in cubic meters (73.5).
    measurement_date : float
        Release date in POSIX seconds, used as the SOFA MeasurementDate
        (no per-measurement date is available).
    """

    name = "sriracha"
    doi = "10.14279/depositonce-23943"
    room_volume = 73.5
    measurement_date = 1755648000.0

    @classmethod
    def get(
        cls,
        scenario: str = "SR1-D",
        dataset_split: str | None = None,
        cache_dir: str | Path | None = None,
        export_dir: str | Path | None = None,
        output_format: str = "pyfar",
    ) -> dict | Path | None:
        """
        scenario : str, optional
            Scenario to download. One of 'SR1', 'SRA1', 'SR1-D', 'SRA1-D',
            'SR2', 'SRA2', 'SR2-D', 'SRA2-D'. Default is 'SR1-D'.
        dataset_split : str or None, optional
            Optional dataset split for full-plane scenarios. One of 'C1',
            'C2', 'C3', 'C4' or None. Dense scenarios (ending in '-D') do not
            have splits. Default is None.

        Returns
        -------
        dict or Path
            For 'pyfar' / 'numpy': dict of in-memory objects.
            For 'sofa' / 'hdf5' / 'raw': Path to file on disk.
        """  # noqa: D205, D403
        return cls()._get(
            scenario=scenario,
            dataset_split=dataset_split,
            cache_dir=cache_dir,
            export_dir=export_dir,
            output_format=output_format,
        )

    def _validate_params(self, **dataset_kwargs) -> None:
        """Validate SRIRACHA-specific parameters.

        Parameters
        ----------
        **dataset_kwargs : dict
            Must contain 'scenario' (one of 'SR1', 'SRA1', 'SR1-D', 'SRA1-D',
            'SR2', 'SRA2', 'SR2-D', 'SRA2-D'). May contain 'dataset_split'
            (one of 'C1', 'C2', 'C3', 'C4', or None). Dense scenarios
            (ending in '-D') cannot be split. ``output_format`` is also
            passed and used to forbid 'raw' for non-dense full-plane
            scenarios.

        Raises
        ------
        ValueError
            If scenario or split is invalid, a dense scenario is combined with
            a split, or 'raw' is requested for a non-dense full plane.
        """
        scenario = dataset_kwargs.get("scenario")
        dataset_split = dataset_kwargs.get("dataset_split")
        output_format = dataset_kwargs.get("output_format")

        if scenario not in ["SR1", "SRA1", "SR1-D", "SRA1-D", "SR2", "SRA2", "SR2-D", "SRA2-D"]:
            raise ValueError("scenario must be one of [SR1, SRA1, SR1-D, SRA1-D, SR2, SRA2, SR2-D, SRA2-D]")
        if dataset_split not in [None, "C1", "C2", "C3", "C4"]:
            raise ValueError("dataset_split must be None or in [C1, C2, C3, C4]")
        if scenario[-1] == "D" and dataset_split is not None:
            raise ValueError("dense datasets do not have splits")
        if output_format == "raw" and scenario and scenario[-1] != "D" and dataset_split is None:
            raise ValueError("raw output_format not supported for non-dense SRIRACHA scenarios without split")

    def _download(self, provider_dir: Path, **dataset_kwargs) -> Path:
        """Download SRIRACHA dataset file(s) to the provider directory.

        For dense scenarios or explicit splits, downloads a single file.
        For non-dense full-plane scenarios, downloads all 4 split files
        and returns the provider directory path.

        Parameters
        ----------
        provider_dir : :class:`pathlib.Path`
            Provider directory (e.g., ``cache/SRIRACHA/provider/``).
        **dataset_kwargs : dict
            Must contain 'scenario'. May contain 'dataset_split'.

        Returns
        -------
        :class:`pathlib.Path`
            Path to the downloaded file (inside provider) or the provider
            directory (for non-dense full-plane scenarios).
        """
        scenario = dataset_kwargs["scenario"]
        split = dataset_kwargs.get("dataset_split")

        # Dense scenario or explicit split -> single-file download
        if scenario.endswith("D") or split is not None:
            fname = self._source_filename(**dataset_kwargs)
            target_file = provider_dir / fname
            logger.info(f"Downloading SRIRACHA scenario {scenario}")
            pup = _pooch_from_doi(self.doi, path=provider_dir)
            _fetch(pup, fname)
            return target_file
        # Non-dense full plane -> download 4 split files; process will then merge them
        logger.info(f"Downloading SRIRACHA scenario {scenario} (4 split files)")
        pup = _pooch_from_doi(self.doi, path=provider_dir)
        for split_file in ["C1", "C2", "C3", "C4"]:
            fname = f"{scenario}-{split_file}.h5"
            _fetch(pup, fname)
        return provider_dir

    def _process(self, provider_artifact: Path, ingest_path: Path, **dataset_kwargs) -> Path:
        """Post-process SRIRACHA file if needed.

        For non-dense full-plane scenarios, merges the 4 downloaded split files
        from the provider directory into a single file in the ingest directory.
        Otherwise promotes the single file to the ingest stage.

        Parameters
        ----------
        provider_artifact : Path
            Path to the downloaded file or the provider directory.
        ingest_path : :class:`pathlib.Path`
            Path to the HDF5 file in the ingest directory.
        **dataset_kwargs : dict
            Must contain 'scenario'. May contain 'dataset_split'.

        Returns
        -------
        Path
            Path to the processed file in the ingest directory.
        """
        scenario = dataset_kwargs["scenario"]
        split = dataset_kwargs.get("dataset_split")

        # Dense scenarios and explicit splits don't need merging
        if scenario.endswith("D") or split is not None:
            return super()._process(provider_artifact, ingest_path, **dataset_kwargs)
        # Non-dense full plane -> merge all 4 split files
        logger.debug("Merging split files")
        return self._merge_split_files(scenario, provider_artifact, ingest_path)

    def _merge_split_files(self, scenario: str, provider_artifact: Path, ingest_path: Path) -> Path:
        """Merges four quadrant HDF5 files into a full-plane file.

        Reads metadata from the first split file in the provider directory,
        allocates output datasets with the full source-grid shape, copies each
        split's measurements into the interleaved grid positions, and deletes
        the provider split files afterwards.

        Parameters
        ----------
        scenario : str
            Scenario name (e.g. 'SR1').
        provider_artifact : Path
            Provider directory where split files are downloaded.
        ingest_path : Path
            Target path in the ingest directory for the merged file.

        Returns
        -------
        Path
            Path to the merged HDF5 file.
        """
        offsets = {"C1": (0, 0), "C2": (0, 1), "C3": (1, 0), "C4": (1, 1)}

        # find split files
        split_files = {}
        for split in offsets:
            fname = f"{scenario}-{split}.h5"
            split_files[split] = provider_artifact / fname

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

        with h5.File(ingest_path, "w") as out:
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

            # delete split files from provider directory
            for f in split_files.values():
                f.unlink()

        logger.debug("Split files merged")

        return ingest_path
