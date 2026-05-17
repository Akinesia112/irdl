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

import numpy as np
import pyfar as pf
import sofar as sf

# Import CACHE_DIR from downloader to maintain consistency


class BaseDataset:
    """Provide common interface for all Dataset implementations.

    Subclasses must define:

    - name : :class:`str`
        Unique identifier for the Dataset.
    - doi : :class:`str`
        Digital Object Identifier for the Dataset.
    - validate_params(dataset_kwargs: dict)
        Validate dataset-specific parameters only (no common params).
    - download(**kwargs) -> Path
        Download and return :class:`pathlib.Path` to raw file.
    - ingest(file_path: Path) -> sofar.Sofa
        Convert raw file to :class:`sofar.Sofa` object.
    - get() @classmethod
        Public entry point with explicit type signature for CLI auto-generation.

    Subclasses may override:

    - validate_output_format(output_format: str, **dataset_kwargs)
        Dataset-specific validation of output_format in context of dataset params.
    - _get(**dataset_kwargs)
        For datasets with special handling (e.g., FABIAN raw output).
    """

    name: str
    doi: str

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
        """Internal implementation of Dataset retrieval.

        Parameters
        ----------
        cache_dir : str
            Cache directory for downloads.
        export_dir : str, optional
            Directory for final output.
        output_format : str
            Output format: 'pyfar', 'numpy', 'hdf5', 'sofa', or 'raw'.
        **dataset_kwargs
            Dataset-specific parameters.

        Returns
        -------
        data : dict or pathlib.Path
            Returned data depends on output_format.
        """
        cache_dir = Path(cache_dir)
        export_dir = Path(export_dir) if export_dir else None

        # Validate common parameters
        if output_format not in ("pyfar", "hdf5", "numpy", "sofa", "raw"):
            raise ValueError("output_format must be one of 'pyfar', 'hdf5', 'numpy', 'sofa', 'raw'")

        # Validate dataset-specific parameters
        self.validate_params(dataset_kwargs)

        # Dataset-specific output_format validation (e.g., SRIRACHA raw restrictions)
        self.validate_output_format(output_format, **dataset_kwargs)

        # Steps 2-3: Get the raw file (download if needed)
        file_path = self._get_file(cache_dir=cache_dir, export_dir=export_dir, **dataset_kwargs)

        # For raw output, return the file directly without processing
        if output_format == "raw":
            return file_path

        # For non-raw: process the file if needed, then ingest and convert
        processed_path = self._process(file_path, **dataset_kwargs)

        # Ingest to SOFA (internal standard)
        sofa = self.ingest(processed_path)

        # Convert to requested output format
        return self._to_output(sofa, output_format, cache_dir, export_dir)

    def validate_output_format(self, output_format: str, **dataset_kwargs) -> None:
        """Validate output_format in the context of dataset-specific parameters.

        Override in subclasses that have output_format restrictions
        (e.g., SRIRACHA blocks raw for non-dense scenarios).

        Parameters
        ----------
        output_format : str
            The output format to validate.
        **dataset_kwargs
            Dataset-specific parameters that may affect validation.
        """
        pass

    def validate_params(self, dataset_kwargs: dict) -> None:
        """Validate dataset-specific parameters only.

        Override in subclass. This method receives only dataset-specific
        parameters (scenario, dataset_split, kind, hato, etc.) — common
        parameters have already been extracted and validated.

        Parameters
        ----------
        dataset_kwargs : dict
            Dataset-specific parameters to validate.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement validate_params()")

    def download(self, **kwargs) -> Path:
        """Download raw files and return :class:`pathlib.Path` to the primary file.

        Override in subclass.

        Parameters
        ----------
        **kwargs : :class:`dict`
            Dataset-specific parameters (scenario, kind, hato, etc.).
            cache_dir and export_dir are NOT in kwargs (handled by get()).

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

    def _get_file(self, cache_dir: Path, export_dir: Path | None, **kwargs) -> Path:
        """Check cache or download file.

        Parameters
        ----------
        cache_dir : :class:`pathlib.Path`
            Base cache directory.
        export_dir : :class:`pathlib.Path` or :class:`None`
            Optional export directory.
        **kwargs : :class:`dict`
            Dataset-specific parameters passed to download().

        Returns
        -------
        file_path : :class:`pathlib.Path`
            Path to the file on disk (either cached or newly downloaded).
        """
        # Construct expected file name based on Dataset (subclass can override)
        file_name = self._construct_file_name(**kwargs)
        file_cache = cache_dir / file_name
        file_export = Path(export_dir) / file_name if export_dir else None

        # Check if file exists in export_dir or cache_dir
        if file_export is not None and file_export.exists():
            return file_export
        if file_cache.exists():
            return file_cache

        # File not cached - download it
        downloaded_path = self.download(cache_dir=cache_dir, **kwargs)

        # Move to export_dir if specified
        if export_dir is not None:
            return self._move_to_export(downloaded_path, export_dir)

        return downloaded_path

    def _process(self, file_path: Path, **kwargs) -> Path:
        """Post-process downloaded file if needed.

        Override in subclass to extract, transform, or otherwise process
        the raw downloaded file before ingestion.

        Parameters
        ----------
        file_path : :class:`pathlib.Path`
            Path to the raw downloaded file.
        **kwargs : :class:`dict`
            Dataset-specific parameters (may be needed for processing decisions).

        Returns
        -------
        file_path : :class:`pathlib.Path`
            Path to the processed file (may be same as input if no processing needed).
        """
        return file_path

    def _construct_file_name(self, **kwargs) -> str:
        """Construct the file name for this Dataset.

        Override in subclass if needed.

        Parameters
        ----------
        **kwargs : :class:`dict`
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

    def _to_output(self, sofa: sf.Sofa, output_format: str, cache_dir: Path, export_dir: Path | None) -> Any:
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
