"""Base Dataset class and conversion utilities for IRDL.

This module provides the BaseDataset class which serves as the common interface
for all Dataset implementations. Each Dataset subclass must implement:

- validate_params()
- download()
- ingest()

The BaseDataset class handles:

- Common parameter extraction
- Path construction
- Cache checking
- Output format conversion from SOFA
"""

from pathlib import Path
from typing import Any

import sofar as sf

# Import CACHE_DIR from downloader to maintain consistency
from irdl.downloader import CACHE_DIR


class BaseDataset:
    """Provide common interface for all Dataset implementations.

    Subclasses must define:

    - name : :class:`str`
        Unique identifier for the Dataset.
    - doi : :class:`str`
        Digital Object Identifier for the Dataset.
    - validate_params()
        Validate dataset-specific parameters.
    - download()
        Download and return :class:`pathlib.Path` to raw file.
    - ingest()
        Convert raw file to :class:`sofar.Sofa` object.
    """

    name: str
    doi: str

    def get(self, **params) -> Any:
        """Retrieve Dataset and return in requested format.

        Parameters
        ----------
        **params : :class:`dict`
            Arbitrary parameters including:
            - cache_dir : :class:`pathlib.Path`
                Override default cache directory.
            - export_dir : :class:`pathlib.Path` or :class:`None`
                Directory for final output.
            - output_format : :class:`str`
                One of "pyfar", "numpy", "hdf5", "sofa".
            - Dataset-specific params (scenario, kind, hato, etc.)

        Returns
        -------
        data : :class:`dict` or :class:`pathlib.Path`
            Returned data depends on output_format:
            - "pyfar" : :class:`dict` of pyfar objects
            - "numpy" : :class:`dict` of numpy arrays
            - "hdf5" : :class:`pathlib.Path` to .h5 file
            - "sofa" : :class:`pathlib.Path` to .sofa file
        """
        # Step 1: Validate all parameters
        self.validate_params(params)

        # Extract common parameters
        cache_dir = Path(params.get("cache_dir", CACHE_DIR))
        export_dir = Path(params.get("export_dir")) if params.get("export_dir") else None
        output_format = params.get("output_format", "pyfar")

        # Steps 2-3: Get the raw file (download if needed)
        file_path = self._get_file(cache_dir=cache_dir, export_dir=export_dir, **params)

        # Ingest to SOFA (internal standard)
        sofa = self.ingest(file_path)

        # Step 7: Convert to requested output format
        return self._to_output(sofa, output_format, cache_dir, export_dir)

    def validate_params(self, params: dict) -> None:
        """Validate dataset-specific parameters.

        Override in subclass.

        Parameters
        ----------
        params : :class:`dict`
            Dataset-specific parameters to validate.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement validate_params()")

    def download(self, **params) -> Path:
        """Download raw files and return :class:`pathlib.Path` to the primary file.

        Override in subclass.

        Parameters
        ----------
        **params : :class:`dict`
            Dataset-specific parameters (scenario, kind, hato, etc.).
            cache_dir and export_dir are NOT in params (handled by get()).

        Returns
        -------
        file_path : :class:`pathlib.Path`
            Path to the downloaded/processed file on disk.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement download()")

    def ingest(self, file_path: Path) -> sf.Sofa:
        """Convert raw file to :class:`sofar.Sofa` object.

        Override in subclass.

        Parameters
        ----------
        file_path : :class:`pathlib.Path`
            Path to the file returned by download().

        Returns
        -------
        sofa : :class:`sofar.Sofa`
            SOFA object representing the Dataset data.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement ingest()")

    def _get_file(self, cache_dir: Path, export_dir: Path | None, **params) -> Path:
        """Check cache or download file.

        Parameters
        ----------
        cache_dir : :class:`pathlib.Path`
            Base cache directory.
        export_dir : :class:`pathlib.Path` or :class:`None`
            Optional export directory.
        **params : :class:`dict`
            Dataset-specific parameters passed to download().

        Returns
        -------
        file_path : :class:`pathlib.Path`
            Path to the file on disk (either cached or newly downloaded).
        """
        # Construct file name based on Dataset (subclass can override)
        file_name = self._construct_file_name(**params)
        file_cache = cache_dir / file_name
        file_export = Path(export_dir) / file_name if export_dir else None

        # Check if file exists in export_dir or cache_dir
        if file_export is not None and file_export.exists():
            return file_export
        if file_cache.exists():
            return file_cache

        # File not cached - download it
        downloaded_path = self.download(cache_dir=cache_dir, export_dir=export_dir, **params)

        # Move to export_dir if specified
        if export_dir is not None:
            return self._move_to_export(downloaded_path, export_dir)

        return downloaded_path

    def _construct_file_name(self, **params) -> str:
        """Construct the file name for this Dataset.

        Override in subclass if needed.

        Parameters
        ----------
        **params : :class:`dict`
            Dataset-specific parameters used to construct the file name.

        Returns
        -------
        :class:`str`
            The constructed file name.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement _construct_file_name()")

    def _move_to_export(self, source: Path, export_dir: Path) -> Path:
        """Move file from source to export_dir.

        Parameters
        ----------
        source : :class:`pathlib.Path`
            Source file path.
        export_dir : :class:`pathlib.Path`
            Target export directory.

        Returns
        -------
        :class:`pathlib.Path`
            Path to the file in export_dir.
        """
        from irdl.utils import _move_to_export_dir

        return _move_to_export_dir(source, str(export_dir))

    def _to_output(self, sofa: sf.Sofa, output_format: str,
                        cache_dir: Path, export_dir: Path | None) -> Any:
        """Convert :class:`sofar.Sofa` to the requested output format.

        Parameters
        ----------
        sofa : :class:`sofar.Sofa`
            SOFA object to convert.
        output_format : :class:`str`
            One of "pyfar", "numpy", "hdf5", "sofa".
        cache_dir : :class:`pathlib.Path`
            Cache directory for file-based outputs.
        export_dir : :class:`pathlib.Path` or :class:`None`
            Optional export directory for file-based outputs.

        Returns
        -------
        :class:`Any`
            Output depends on output_format:
            - "pyfar" : :class:`dict` of pyfar objects
            - "numpy" : :class:`dict` of numpy arrays
            - "hdf5" : :class:`pathlib.Path` to .h5 file
            - "sofa" : :class:`pathlib.Path` to .sofa file
        """
        if output_format == "pyfar":
            return self._to_pyfar(sofa)
        elif output_format == "numpy":
            return self._to_numpy(sofa)
        elif output_format == "sofa":
            return self._to_sofa(sofa, cache_dir, export_dir)
        elif output_format == "hdf5":
            return self._to_hdf5(sofa, cache_dir, export_dir)
        else:
            raise ValueError(f"Unknown output_format: {output_format}")

    def _to_pyfar(self, sofa: sf.Sofa) -> dict:
        """Convert :class:`sofar.Sofa` to :class:`dict` of pyfar objects.

        Parameters
        ----------
        sofa : :class:`sofar.Sofa`
            SOFA object to convert.

        Returns
        -------
        :class:`dict`
            Dictionary with keys:
            - "impulse_response" : :class:`pyfar.Signal`
            - "source_coordinates" : :class:`pyfar.Coordinates`
            - "receiver_coordinates" : :class:`pyfar.Coordinates`
        """
        import pyfar as pf

        return {
            "impulse_response": pf.Signal(sofa.Data_IR, sampling_rate=sofa.Data_SamplingRate),
            "source_coordinates": pf.Coordinates(*sofa.SourcePosition.T),
            "receiver_coordinates": pf.Coordinates(*sofa.ReceiverPosition.T),
        }

    def _to_numpy(self, sofa: sf.Sofa) -> dict:
        """Convert :class:`sofar.Sofa` to :class:`dict` of numpy arrays.

        Parameters
        ----------
        sofa : :class:`sofar.Sofa`
            SOFA object to convert.

        Returns
        -------
        :class:`dict`
            Dictionary with keys:
            - "impulse_response" : :class:`numpy.ndarray`
            - "source_coordinates" : :class:`numpy.ndarray`
            - "receiver_coordinates" : :class:`numpy.ndarray`
            - "sampling_rate" : :class:`float`
        """
        import numpy as np

        return {
            "impulse_response": np.array(sofa.Data_IR),
            "source_coordinates": np.array(sofa.SourcePosition),
            "receiver_coordinates": np.array(sofa.ReceiverPosition),
            "sampling_rate": float(sofa.Data_SamplingRate),
        }

    def _to_sofa(self, sofa: sf.Sofa, cache_dir: Path, export_dir: Path | None) -> Path:
        """Write :class:`sofar.Sofa` to file and return :class:`pathlib.Path`.

        Parameters
        ----------
        sofa : :class:`sofar.Sofa`
            SOFA object to write.
        cache_dir : :class:`pathlib.Path`
            Cache directory (used if export_dir is None).
        export_dir : :class:`pathlib.Path` or :class:`None`
            Optional export directory.

        Returns
        -------
        :class:`pathlib.Path`
            Path to the written SOFA file.
        """
        import sofar as sf

        # Generate unique file name
        file_name = f"{self.name}.sofa"
        if export_dir:
            path = export_dir / file_name
        else:
            path = cache_dir / self.name / file_name
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write_sofa(str(path), sofa)
        return path

    def _to_hdf5(self, sofa: sf.Sofa, cache_dir: Path, export_dir: Path | None) -> Path:
        """Convert :class:`sofar.Sofa` to HDF5 file and return :class:`pathlib.Path`.

        TODO: Implement proper conversion to match MIRACLE/SRIRACHA structure.
        For now, this is a placeholder that writes a basic HDF5 file.

        Parameters
        ----------
        sofa : :class:`sofar.Sofa`
            SOFA object to convert.
        cache_dir : :class:`pathlib.Path`
            Cache directory (used if export_dir is None).
        export_dir : :class:`pathlib.Path` or :class:`None`
            Optional export directory.

        Returns
        -------
        :class:`pathlib.Path`
            Path to the written HDF5 file.
        """
        import h5py as h5

        file_name = f"{self.name}.h5"
        if export_dir:
            path = export_dir / file_name
        else:
            path = cache_dir / self.name / file_name
        path.parent.mkdir(parents=True, exist_ok=True)

        with h5.File(path, "w") as f:
            # Create data group
            data_group = f.create_group("data")
            data_group.create_dataset("impulse_response", data=sofa.Data_IR)

            # Create location group
            loc_group = data_group.create_group("location")
            loc_group.create_dataset("source", data=sofa.SourcePosition)
            loc_group.create_dataset("receiver", data=sofa.ReceiverPosition)

            # Create metadata group
            meta_group = f.create_group("metadata")
            meta_group.create_dataset("sampling_rate", data=sofa.Data_SamplingRate)

            # Add other metadata if present
            if hasattr(sofa, "Data_Temperature"):
                meta_group.create_dataset("temperature", data=sofa.Data_Temperature)
            if hasattr(sofa, "Data_Humidity"):
                meta_group.create_dataset("humidity", data=sofa.Data_Humidity)

        return path
