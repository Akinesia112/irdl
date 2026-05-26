"""Base Dataset class and conversion utilities for IRDL.

This module provides the BaseDataset abstract base class which serves as the common interface
for all Dataset implementations. Each Dataset subclass must implement:

- validate_params()
- download()
- ingest()
- _source_filename()

The BaseDataset class handles:

- Common parameter extraction
- Path construction
- Cache checking
- Output format conversion from SOFA
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pyfar as pf
import sofar as sf

from irdl.logger import logger


class BaseDataset(ABC):
    """Abstract base class providing common interface for all Dataset implementations.

    Subclasses must implement the following abstract methods:

    Attributes
    ----------
    name : str
        Unique identifier for the Dataset.
    doi : str
        Digital Object Identifier for the Dataset.

    Methods
    -------
    validate_params(**dataset_kwargs)
        Validate dataset-specific parameters (including output_format).
    download(**kwargs) -> Path
        Download and return Path to raw file.
    ingest(file_path: Path) -> sofar.Sofa
        Convert raw file to sofar.Sofa object.
    _source_filename(**kwargs) -> str
        Construct the raw input filename with extension.
    get() @classmethod
        Public entry point with explicit type signature for CLI auto-generation.
    """

    name: str
    doi: str

    # Default docstring prefix for all get() classmethods
    _get_doc_prefix = """Download {name} dataset.

Parameters
----------
cache_dir : str
    Cache directory for downloads.
export_dir : str, optional
    Directory for final output. Stays in cache_dir if not specified.
output_format : str
    Output format: 'pyfar', 'numpy', 'hdf5', 'sofa', or 'raw'.
"""

    def __init_subclass__(cls, **kwargs):
        """Initialize subclass with automatic docstring composition for get() classmethod."""
        super().__init_subclass__(**kwargs)
        # Automatically compose docstrings for get() classmethod
        if hasattr(cls, "get") and hasattr(cls, "name") and hasattr(cls, "doi"):
            # Get the underlying function of the classmethod
            get_func = cls.get.__func__
            # Get the first line of the class docstring for the summary
            class_doc = cls.__doc__ or ""
            doc_lines = class_doc.strip().split("\n") if class_doc.strip() else []
            summary_line = doc_lines[0] if doc_lines else ""
            # Construct DOI line from cls.doi attribute
            doi_url = f"https://doi.org/{cls.doi}"
            doi_cli_line = f"DOI: {doi_url}"
            # Format prefix with class attributes
            prefix = BaseDataset._get_doc_prefix.format(name=cls.name, doi=cls.doi)
            # If class has a docstring with a summary, replace the first line of prefix
            if summary_line:
                # Split prefix into lines and replace the first line
                prefix_lines = prefix.split("\n")
                prefix_lines[0] = summary_line
                # Insert DOI line after the summary
                prefix_lines.insert(1, "")
                prefix_lines.insert(2, doi_cli_line)
                prefix = "\n".join(prefix_lines)
            # Get subclass-specific docstring (from the base class _get method)
            suffix = get_func.__doc__ or ""
            # Combine: prefix + suffix
            full_doc = prefix
            if suffix:
                full_doc += suffix
            get_func.__doc__ = full_doc

    def _get(
        self,
        cache_dir: Path | str,
        export_dir: Path | str | None,
        output_format: str,
        **dataset_kwargs,
    ) -> Any:
        """Internal implementation of Dataset retrieval.

        Parameters
        ----------
        cache_dir : :class:`pathlib.Path` or str
            Cache directory for downloads.
        export_dir : :class:`pathlib.Path` or str or None
            Directory for final output. Default is None (stays in cache_dir).
        output_format : str
            Output format: 'pyfar', 'numpy', 'hdf5', 'sofa', or 'raw'.
        **dataset_kwargs : dict
            Dataset-specific parameters.

        Returns
        -------
        dict or :class:`pathlib.Path`
            For 'pyfar' / 'numpy': a dict of in-memory objects.
            For 'sofa' / 'hdf5' / 'raw': a :class:`pathlib.Path` to the file on disk.
        """
        cache_dir = Path(cache_dir)
        export_dir = Path(export_dir) if export_dir else None

        # Validate common parameters
        if output_format not in ("pyfar", "hdf5", "numpy", "sofa", "raw"):
            raise ValueError("output_format must be one of 'pyfar', 'hdf5', 'numpy', 'sofa', 'raw'")

        # Validate dataset-specific parameters (including output_format)
        logger.debug(f"Validating parameters for {self.name}")
        self._validate_params(output_format=output_format, **dataset_kwargs)

        # Early exit if output file already exists
        output_path = self._output_path(output_format, cache_dir, export_dir, **dataset_kwargs)
        if output_path is not None and output_path.exists():
            logger.info(f"Output file already exists at {output_path}, skipping download and conversion.")
            return output_path

        # path to cache file
        file_path = self._input_path(cache_dir, None, **dataset_kwargs)
        if file_path.exists():
            logger.info(f"Cache file already exists at {file_path}, skipping download.")
        else:
            logger.info(f"Getting {self.name.upper()} dataset to {file_path}.")
            file_path = self._download(file_path, **dataset_kwargs)

        # return raw file if requested
        if output_format == "raw":
            if export_dir is None:
                logger.debug(f"Returning raw file at {file_path}.")
                return file_path
            else:
                return self._move_to_export(file_path, export_dir)
        else:
            # Process the file if needed (e.g., extraction, merging)
            logger.debug(f"Processing {file_path}")
            processed_path = self._process(file_path, cache_dir=cache_dir, export_dir=export_dir, **dataset_kwargs)

        # Ingest to SOFA (internal standard)
        logger.debug(f"Ingesting {processed_path} to SOFA format. Nom nom ...")
        sofa = self._ingest(processed_path)

        # We check for correctness here. This gives instant feedback when adding new datasets.
        try:
            sofa.verify(issue_handling="raise")
            with logger.as_stdout:
                sofa.upgrade_convention()
        except ValueError as e:
            logger.error(
                f"SOFA convention not satisfied!\n{e}\nSee https://sofar.readthedocs.io/en/stable/resources/conventions.html#conventions for details."
            )
            return

        # Convert to requested output format
        logger.debug(f"Converting to {output_format} format")
        return self._to_output(sofa, output_format, output_path)

    @abstractmethod
    def _validate_params(self, **dataset_kwargs) -> None:
        """Validate dataset-specific parameters.

        Override in subclass. This method receives dataset-specific parameters
        plus ``output_format`` (so subclasses can forbid invalid output_format /
        dataset-parameter combinations).

        Parameters
        ----------
        **dataset_kwargs : dict
            Dataset-specific parameters to validate, including ``output_format``.

        Raises
        ------
        ValueError
            If any parameter is invalid.
        """

    @abstractmethod
    def _download(self, target_path: Path, **kwargs) -> Path:
        """Download raw files and return Path to the primary file.

        Override in subclass.

        Parameters
        ----------
        target_path : :class:`pathlib.Path`
            Target path where the file should be downloaded.
        **kwargs : dict
            Dataset-specific parameters (scenario, kind, hato, etc.).
            cache_dir and export_dir are NOT in kwargs (handled by _get()).

        Returns
        -------
        file_path : :class:`pathlib.Path`
            Path to the downloaded/processed file on disk.
        """

    @abstractmethod
    def _ingest(self, file_path: Path) -> sf.Sofa:
        """Convert raw file to sofar.Sofa object.

        Override in subclass.

        Parameters
        ----------
        file_path : :class:`pathlib.Path`
            Path to the file returned by _get_file().

        Returns
        -------
        sofa : :class:`sofar.Sofa`
            SOFA object representing the Dataset data.
        """

    @abstractmethod
    def _source_filename(self, **kwargs) -> str:
        """Construct the raw input filename with extension for the dataset.

        Override in subclass.

        Parameters
        ----------
        **kwargs : dict
            Dataset-specific parameters used to construct the filename.

        Returns
        -------
        str
            The raw input filename including extension (e.g., "A1.h5",
            "FABIAN_HRIR_measured_HATO_0.sofa").
        """

    def _input_path(self, cache_dir: Path, export_dir: Path | None, **kwargs) -> Path:
        """Return the full path to the raw input file.

        Parameters
        ----------
        cache_dir : Path
            Cache directory.
        export_dir : Path or None
            Optional export directory; takes priority over cache_dir.
        **kwargs : dict
            Dataset-specific parameters passed to _source_filename().

        Returns
        -------
        Path
            Full path to the raw input file under '<base>/<DATASET_NAME>/'.
        """
        base = (export_dir if export_dir is not None else cache_dir) / self.name.upper()
        return base / self._source_filename(**kwargs)

    def _output_path(self, output_format: str, cache_dir: Path, export_dir: Path | None, **kwargs) -> Path | None:
        """Return the canonical Path where a file-based output would be written.

        Returns None for in-memory formats ('pyfar', 'numpy'). Uses _source_filename
        to construct the base filename, then replaces the extension based on output_format.

        Parameters
        ----------
        output_format : str
            One of 'pyfar', 'numpy', 'hdf5', 'sofa', 'raw'.
        cache_dir : Path
            Cache directory.
        export_dir : Path or None
            Optional export directory; takes priority over cache_dir.
        **kwargs : dict
            Dataset-specific parameters used to construct the filename.

        Returns
        -------
        Path or None
            Canonical output path under '<base>/<DATASET_NAME>/', or None for
            in-memory formats.
        """
        source_filename = Path(self._source_filename(**kwargs))

        # Determine extension based on output format
        match output_format:
            case "numpy" | "pyfar":
                return None
            case "sofa":
                suff = ".sofa"
            case "hdf5":
                suff = ".h5"
            case "raw":
                suff = source_filename.suffix

        base = (export_dir if export_dir is not None else cache_dir) / self.name.upper()
        return (base / source_filename.stem).with_suffix(suff)

    def _process(self, file_path: Path, **kwargs) -> Path:
        """Post-process downloaded file if needed.

        Override in subclass to extract, transform, or otherwise process
        the raw downloaded file before ingestion.

        Parameters
        ----------
        file_path : Path
            Path to the raw downloaded file.
        **kwargs : dict
            Dataset-specific parameters (may be needed for processing decisions).

        Returns
        -------
        file_path : Path
            Path to the processed file (may be same as input if no processing needed).
        """
        return file_path

    def _move_to_export(self, source: Path, export_dir: Path) -> Path:
        """Move file from source to export_dir.

        Parameters
        ----------
        source : Path
            Source file path.
        export_dir : Path
            Target export directory.

        Returns
        -------
        Path
            Path to the file in export_dir.
        """
        import shutil

        target = Path(export_dir) / source.name
        # file exists already in export_dir
        if target.exists():
            return target
        # move file from source to export_dir
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(source, target)

        return target

    def _to_output(self, sofa: sf.Sofa, output_format: str, output_path: Path | None) -> Any:
        """Convert sofar.Sofa to the requested output format.

        Parameters
        ----------
        sofa : :class:`sofar.Sofa`
            SOFA object to convert.
        output_format : str
            One of "pyfar", "numpy", "hdf5", "sofa".
        output_path : :class:`pathlib.Path` or None
            Path where file-based outputs should be written.

        Returns
        -------
        dict or :class:`pathlib.Path`
            Output depends on output_format:
            - "pyfar" : dict of :class:`pyfar.Signal` and :class:`pyfar.Coordinates` objects
            - "numpy" : dict of :class:`numpy.ndarray` arrays
            - "hdf5" : :class:`pathlib.Path` to .h5 file
            - "sofa" : :class:`pathlib.Path` to .sofa file
        """
        if output_format == "pyfar":
            return self._to_pyfar(sofa)
        elif output_format == "numpy":
            return self._to_numpy(sofa)
        elif output_format == "sofa":
            return self._to_sofa(sofa, output_path)
        elif output_format == "hdf5":
            return self._to_hdf5(sofa, output_path)
        else:
            raise ValueError(f"Unknown output_format: {output_format}")

    def _to_pyfar(self, sofa: sf.Sofa) -> dict:
        """Convert sofar.Sofa to dict of pyfar objects.

        Parameters
        ----------
        sofa : :class:`sofar.Sofa`
            SOFA object to convert.

        Returns
        -------
        dict
            Dictionary with keys:
            - "impulse_response" : :class:`pyfar.Signal`
            - "source_coordinates" : :class:`pyfar.Coordinates`
            - "receiver_coordinates" : :class:`pyfar.Coordinates`
        """
        return dict(
            zip(
                ("impulse_response", "source_coordinates", "receiver_coordinates"),
                pf.io.convert_sofa(sofa),
                strict=True,
            )
        )

    def _to_numpy(self, sofa: sf.Sofa) -> dict:
        """Convert sofar.Sofa to dict of numpy arrays.

        Parameters
        ----------
        sofa : :class:`sofar.Sofa`
            SOFA object to convert.

        Returns
        -------
        dict
            Dictionary with keys:
            - "impulse_response" : :class:`numpy.ndarray`
            - "source_coordinates" : :class:`numpy.ndarray`
            - "receiver_coordinates" : :class:`numpy.ndarray`
            - "sampling_rate" : float
        """
        return {
            "impulse_response": np.array(sofa.Data_IR),
            "source_coordinates": np.array(sofa.SourcePosition),
            "receiver_coordinates": np.array(sofa.ReceiverPosition),
            "sampling_rate": float(sofa.Data_SamplingRate),
        }

    def _to_sofa(self, sofa: sf.Sofa, output_path: Path) -> Path:
        """Write sofar.Sofa to file and return Path.

        Parameters
        ----------
        sofa : :class:`sofar.Sofa`
            SOFA object to write.
        output_path : :class:`pathlib.Path`
            Path where the .sofa file should be written.

        Returns
        -------
        :class:`pathlib.Path`
            Path to the written SOFA file.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write_sofa(str(output_path), sofa)
        return output_path

    def _to_hdf5(self, sofa: sf.Sofa, output_path: Path) -> Path:
        """Convert sofar.Sofa to HDF5 file and return Path.

        Parameters
        ----------
        sofa : :class:`sofar.Sofa`
            SOFA object to convert.
        output_path : :class:`pathlib.Path`
            Path where the .h5 file should be written.

        Returns
        -------
        :class:`pathlib.Path`
            Path to the written HDF5 file.
        """
        import h5py as h5

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with h5.File(output_path, "w") as f:
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
            if hasattr(sofa, "RoomTemperature"):
                meta_group.create_dataset("temperature", data=sofa.RoomTemperature)
            elif hasattr(sofa, "Data_Temperature"):
                meta_group.create_dataset("temperature", data=sofa.Data_Temperature)
            if hasattr(sofa, "Data_Humidity"):
                meta_group.create_dataset("humidity", data=sofa.Data_Humidity)

        return output_path
