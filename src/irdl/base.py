"""Base Dataset class and conversion utilities for IRDL.

Provides the BaseDataset class that defines the common interface and pipeline
shared by all dataset implementations. Subclasses implement dataset-specific
logic for parameter validation, file retrieval, ingestion, and output naming.
"""

from pathlib import Path
from typing import Any

import h5py as h5
import numpy as np
import pyfar as pf
import sofar as sf


class BaseDataset:
    """Common interface and pipeline for all dataset implementations.

    Subclasses must define:

    - name : str
        Unique identifier for the dataset.
    - doi : str
        Digital Object Identifier for the dataset.
    - raw_format : str
        Native file format of the publisher's raw files (e.g. 'hdf5' for ISTA,
        'sofa' for FABIAN). Used to short-circuit the pipeline when the
        requested output format matches the raw format.
    - validate_params(**dataset_kwargs)
        Validate dataset-specific parameters.
    - _get_file(cache_dir, export_dir, **kwargs) -> Path
        Fetch and process the raw file; return the path ready for ingest().
    - ingest(file_path) -> sofar.Sofa
        Convert the raw file into a SOFA object.
    - _output_path(output_format, cache_dir, export_dir, **kwargs) -> Path or None
        Canonical path where output is written; None for in-memory formats.
    - get() @classmethod
        Public entry point with an explicit signature for CLI auto-generation.
    """

    name: str
    doi: str
    raw_format: str

    # Default docstring prefix for all get() classmethods
    _get_doc_prefix = """Download {name} dataset.

DOI: {doi}

Parameters
----------
cache_dir : str
    Cache directory for downloads. Default: user cache directory.
export_dir : str, optional
    Directory for final output. Default: None (stays in cache_dir).
output_format : str
    Output format: 'pyfar', 'numpy', 'hdf5', 'sofa', or 'raw'.
"""

    def __init_subclass__(cls, **kwargs):
        """Compose the get() docstring when a subclass is defined."""
        super().__init_subclass__(**kwargs)
        # Automatically compose docstrings for get() classmethod
        if hasattr(cls, "get") and hasattr(cls, "name") and hasattr(cls, "doi"):
            # Get the underlying function of the classmethod
            get_func = cls.get.__func__
            # Format prefix with class attributes
            prefix = BaseDataset._get_doc_prefix.format(name=cls.name, doi=cls.doi)
            # Get subclass-specific docstring
            suffix = get_func.__doc__ or ""
            # Combine: prefix + suffix
            full_doc = prefix
            if suffix:
                if not full_doc.endswith("\n\n"):
                    full_doc += "\n\n"
                full_doc += suffix
            get_func.__doc__ = full_doc

    def _get(self, *, cache_dir: str, export_dir: str | None, output_format: str, **dataset_kwargs) -> Any:
        """Execute the full dataset retrieval pipeline.

        Validates parameters, returns an existing output if found, otherwise
        fetches and processes the raw file, ingests it to SOFA, and converts
        to the requested output format.

        Parameters
        ----------
        cache_dir : str
            Cache directory for downloads and intermediate files.
        export_dir : str or None
            Optional directory for final output. If None, files stay in
            cache_dir.
        output_format : str
            One of 'pyfar', 'numpy', 'hdf5', 'sofa', 'raw'.
        **dataset_kwargs
            Dataset-specific parameters.

        Returns
        -------
        dict or Path
            For 'pyfar' / 'numpy': a dict of in-memory objects.
            For 'sofa' / 'hdf5' / 'raw': a Path to the file on disk.
        """
        cache_dir = Path(cache_dir)
        export_dir = Path(export_dir) if export_dir else None

        # Validate common parameters
        if output_format not in ("pyfar", "hdf5", "numpy", "sofa", "raw"):
            raise ValueError("output_format must be one of 'pyfar', 'hdf5', 'numpy', 'sofa', 'raw'")

        # Validate dataset-specific parameters
        self.validate_params(output_format=output_format, **dataset_kwargs)

        # define output_path and check if (file-based) output exists already
        output_path = self._output_path(output_format, cache_dir, export_dir, **dataset_kwargs)
        if output_path is not None and output_path.exists():
            return output_path

        # Get the raw file (download if needed)
        file_path = self._get_file(cache_dir, export_dir, **dataset_kwargs)

        # For raw output, return the file directly without processing
        if output_format == "raw" or output_format == self.raw_format:
            return file_path

        # Ingest to SOFA (internal standard)
        sofa = self.ingest(file_path)

        # Convert to requested output format
        return self._to_output(sofa, output_format, output_path)

    def _output_path(self, output_format: str, cache_dir: Path, export_dir: Path | None, **kwargs) -> Path | None:
        """Return the canonical Path where a file-based output would be written.

        Returns None for in-memory formats ('pyfar', 'numpy'). Subclasses
        override to encode dataset-specific naming for file-based formats.

        Parameters
        ----------
        output_format : str
            One of 'pyfar', 'numpy', 'hdf5', 'sofa', 'raw'.
        cache_dir : Path
            Cache directory.
        export_dir : Path or None
            Optional export directory; takes priority over cache_dir.
        **kwargs
            Dataset-specific parameters used to construct the file name.

        Returns
        -------
        Path or None
            Canonical output path, or None for in-memory formats.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement output_path)")

    def validate_params(self, **dataset_kwargs) -> None:
        """Validate dataset-specific parameters.

        Override in subclass. Receives the dataset-specific parameters plus
        ``output_format`` (so subclasses can forbid invalid output_format /
        dataset-parameter combinations).

        Parameters
        ----------
        **dataset_kwargs
            Dataset-specific parameters to validate, plus ``output_format``.

        Raises
        ------
        ValueError
            If any parameter is invalid.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement validate_params()")

    def _get_file(self, cache_dir, export_dir, **kwargs) -> Path:
        """Return the path to a file ready for ingest().

        Subclass implements the input-side flow: check cache, download what is
        missing, perform any merging or splitting, return the final path.

        Parameters
        ----------
        cache_dir : Path
            Cache directory for downloads.
        export_dir : Path or None
            Optional directory for final output; takes priority over cache_dir
            for writes.
        **kwargs
            Dataset-specific parameters.

        Returns
        -------
        Path
            Path to the file ready to be passed to ingest().
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement _get_file()")

    def ingest(self, file_path: Path) -> sf.Sofa:
        """Convert a raw file into a sofar.Sofa object.

        Override in subclass.

        Parameters
        ----------
        file_path : Path
            Path to the file returned by _get_file().

        Returns
        -------
        sofar.Sofa
            SOFA object representing the dataset.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement ingest()")

    def _to_output(self, sofa: sf.Sofa, output_format: str, output_path: Path) -> Any:
        """Dispatch a sofar.Sofa object to the requested output format.

        Parameters
        ----------
        sofa : sofar.Sofa
            SOFA object to convert.
        output_format : str
            One of 'pyfar', 'numpy', 'hdf5', 'sofa'.
        output_path : Path
            Where to write file-based outputs ('sofa', 'hdf5'). Ignored for
            in-memory formats ('pyfar', 'numpy').

        Returns
        -------
        dict or Path
            'pyfar' / 'numpy': dict of in-memory objects.
            'sofa' / 'hdf5': Path to the written file.
        """
        if output_format == "pyfar":
            return self._to_pyfar(sofa)
        if output_format == "numpy":
            return self._to_numpy(sofa)
        if output_format == "sofa":
            return self._to_sofa(sofa, output_path)
        if output_format == "hdf5":
            return self._to_hdf5(sofa, output_path)

    def _to_pyfar(self, sofa: sf.Sofa) -> dict:
        """Convert a sofar.Sofa object into a dict of pyfar objects.

        Parameters
        ----------
        sofa : sofar.Sofa
            SOFA object to convert.

        Returns
        -------
        dict
            Keys:
            - 'impulse_response' : pyfar.Signal
            - 'source_coordinates' : pyfar.Coordinates
            - 'receiver_coordinates' : pyfar.Coordinates
        """
        receiver = np.asarray(sofa.ReceiverPosition).squeeze()  # (R, C, I) -> (R, 3)
        source = np.asarray(sofa.SourcePosition)  # (M, C)
        ir = np.asarray(sofa.Data_IR).squeeze()
        sr = float(np.asarray(sofa.Data_SamplingRate).flat[0])

        return {
            "impulse_response": pf.Signal(ir, sampling_rate=sr),
            "source_coordinates": pf.Coordinates(source[:, 0], source[:, 1], source[:, 2]),
            "receiver_coordinates": pf.Coordinates(receiver[:, 0], receiver[:, 1], receiver[:, 2]),
        }

    def _to_numpy(self, sofa: sf.Sofa) -> dict:
        """Convert a sofar.Sofa object into a dict of numpy arrays.

        Parameters
        ----------
        sofa : sofar.Sofa
            SOFA object to convert.

        Returns
        -------
        dict
            Keys:
            - 'impulse_response' : numpy.ndarray
            - 'source_coordinates' : numpy.ndarray
            - 'receiver_coordinates' : numpy.ndarray
            - 'sampling_rate' : float
        """
        return {
            "impulse_response": np.array(sofa.Data_IR),
            "source_coordinates": np.array(sofa.SourcePosition),
            "receiver_coordinates": np.array(sofa.ReceiverPosition),
            "sampling_rate": float(np.asarray(sofa.Data_SamplingRate).flat[0]),
        }

    def _to_sofa(self, sofa: sf.Sofa, output_path: Path) -> Path:
        """Write a sofar.Sofa object to a .sofa file.

        Parameters
        ----------
        sofa : sofar.Sofa
            SOFA object to write.
        output_path : Path
            Path where the .sofa file is written.

        Returns
        -------
        Path
            Path to the written .sofa file.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write_sofa(str(output_path), sofa)
        return output_path

    def _to_hdf5(self, sofa: sf.Sofa, output_path: Path) -> Path:
        """Write a sofar.Sofa object as an HDF5 file.

        Parameters
        ----------
        sofa : sofar.Sofa
            SOFA object to write.
        output_path : Path
            Path where the .h5 file is written.

        Returns
        -------
        Path
            Path to the written .h5 file.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with h5.File(output_path, "w") as f:
            data_group = f.create_group("data")
            data_group.create_dataset("impulse_response", data=sofa.Data_IR)
            loc_group = data_group.create_group("location")
            loc_group.create_dataset("source", data=sofa.SourcePosition)
            loc_group.create_dataset("receiver", data=sofa.ReceiverPosition)
            meta_group = f.create_group("metadata")
            meta_group.create_dataset("sampling_rate", data=sofa.Data_SamplingRate)

            # Add other metadata if present
            if hasattr(sofa, "RoomTemperature"):
                meta_group.create_dataset("temperature", data=sofa.RoomTemperature)

        return output_path

    def _lookup(
        self,
        file_name: str,
        cache_dir: Path,
        export_dir: Path | None,
    ) -> Path | None:
        """Return the path if the file exists in export_dir or cache_dir, else None.

        Searches export_dir first, falling back to cache_dir.

        Parameters
        ----------
        file_name : str
            File name to search for.
        cache_dir : Path
            Cache directory to search.
        export_dir : Path or None
            Optional export directory to search first.

        Returns
        -------
        Path or None
            Path to the existing file, or None if not found.
        """
        if export_dir is not None:
            p = export_dir / file_name
            if p.exists():
                return p
        p = cache_dir / file_name
        if p.exists():
            return p
        return None
