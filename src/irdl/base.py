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

import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

import h5py as h5
import numpy as np
import pyfar as pf
import sofar as sf

from irdl.downloader import IRDL_CACHE_DIR
from irdl.logging import logger
from irdl.utils import _fits_in_memory


class BaseDataset(ABC):
    """Abstract base class providing common interface for all Dataset implementations.

    Attributes
    ----------
    name : str
        Unique identifier for the Dataset.
    doi : str
        Digital Object Identifier for the Dataset.

    Methods
    -------
    _validate_params(**dataset_kwargs)
        Validate dataset-specific parameters (including output_format).
    _download(**dataset_kwargs) -> Path
        Download and return Path to raw file.
    _ingest(ingest_path: Path) -> sofar.Sofa
        Convert raw file to sofar.Sofa object.
    _source_filename(**dataset_kwargs) -> str
        Construct the raw input filename with extension.
    get() @classmethod
        Public entry point. Uses explicit type signature for CLI auto-generation.
    """

    name: str
    doi: str

    # Default docstring prefix for all get() classmethods
    _get_doc_prefix = """Download {name} dataset.

Parameters
----------
cache_dir : str
    Cache directory for downloads. Defaults is the OS user cache directory.
    This default can be overridden by setting `IRDL_CACHE_DIR` environment variable.
export_dir : str, optional
    Directory for final output. If specified, the data will be exported to <export_dir/{name}/>. Else, it remains in
    <cache_dir/output/>.
output_format : str
    Output format: 'pyfar', 'numpy', 'hdf5', 'sofa', or 'raw'.
"""

    def __init_subclass__(cls, **dataset_kwargs) -> None:
        """Initialize subclass with automatic docstring composition for get() classmethod."""
        super().__init_subclass__(**dataset_kwargs)
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
            prefix = BaseDataset._get_doc_prefix.format(name=cls.name.upper(), doi=cls.doi)
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
        cache_dir: Path | str | None,
        export_dir: Path | str | None,
        output_format: str,
        **dataset_kwargs,
    ) -> dict | Path | None:
        """Internal implementation of Dataset retrieval.

        Parameters
        ----------
        cache_dir : :class:`pathlib.Path` or str or None
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
        # Validate common parameters
        if output_format not in ("pyfar", "hdf5", "numpy", "sofa", "raw"):
            raise ValueError("output_format must be one of 'pyfar', 'hdf5', 'numpy', 'sofa', 'raw'")

        # Validate dataset-specific parameters (including output_format)
        logger.debug(f"Validating parameters for {self.name}")
        self._validate_params(output_format=output_format, **dataset_kwargs)

        # Set up and sanitize path variables
        cache_dir = (IRDL_CACHE_DIR if cache_dir is None else Path(cache_dir)) / self.name.upper()
        export_dir = None if export_dir is None else Path(export_dir)
        output_dir = cache_dir / "output" if export_dir is None else export_dir / self.name.upper()
        provider_dir = cache_dir / "provider"
        source_filename = self._source_filename(**dataset_kwargs)
        output_path = self._output_path(output_dir, source_filename, output_format)
        ingest_path = cache_dir / "ingest" / source_filename

        # Special handling for raw output format
        if output_format == "raw":
            provider_artifact = self.download(provider_dir, **dataset_kwargs)
            if export_dir is None:
                return provider_artifact
            else:
                return self._export_raw(provider_artifact, export_dir)

        # Early exit if output file already exists (not applicable for raw format, handled above)
        if output_path is not None and output_path.exists():
            logger.info(f"Output file already exists at {output_path}, skipping download and conversion.")
            return output_path

        # Check if ingest-ready file already exists
        if ingest_path.exists():
            logger.info(f"Ingestible file already exists at {ingest_path}, skipping download and processing.")
        else:
            # Download to provider directory
            provider_artifact = self.download(provider_dir, **dataset_kwargs)
            logger.debug(f"Processing {provider_artifact} to {ingest_path}")
            ingest_path = self.process(provider_artifact, ingest_path, **dataset_kwargs)

        # Ingest to SOFA (internal standard)
        if _fits_in_memory(ingest_path):
            logger.debug(f"Ingesting {ingest_path} to SOFA format. Nom nom ...")
            sofa = self._ingest(ingest_path)
        else:
            logger.warning(f"Not enough memory for conversion, returning {ingest_path} instead ...")
            return ingest_path

        # Check for correct SOFA conventions. This gives instant feedback when adding new datasets.
        try:
            sofa.verify(issue_handling="raise")
            with logger.as_stdout:
                sofa.upgrade_convention()
        except ValueError as e:
            logger.error(
                f"SOFA convention not satisfied!\n{e}\n"
                "See https://sofar.readthedocs.io/en/stable/resources/conventions.html#conventions for details."
            )
            return

        logger.debug(f"Converting to {output_format} format")
        return self._to_output(sofa, output_format, ingest_path, output_path)

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

    def download(self, provider_dir: Path, **dataset_kwargs) -> Path:
        """Download raw files and return Path to the primary artifact.

        This method wraps _download to enforce provider_dir existence for all subclasses.

        Parameters
        ----------
        provider_dir : :class:`pathlib.Path`
            Target path where the file(s) should be downloaded to.
            For single-file providers, this may be the file path itself.
            For multi-file providers, this may be a directory where files are placed.
        **dataset_kwargs : dict
            Dataset-specific parameters.

        Returns
        -------
        provider_artifact : :class:`pathlib.Path`
            Path to the downloaded artifact on disk (file or directory).
        """
        provider_dir.mkdir(exist_ok=True, parents=True)
        return self._download(provider_dir, **dataset_kwargs)

    @abstractmethod
    def _download(self, provider_dir: Path, **dataset_kwargs) -> Path:
        """Concrete download logic. Override in subclass."""

    @abstractmethod
    def _ingest(self, ingest_path: Path) -> sf.Sofa:
        """Convert raw file to sofar.Sofa object.

        Override in subclass.

        Parameters
        ----------
        ingest_path : :class:`pathlib.Path`
            Path to the ingest-ready file in the ``ingest/`` subdirectory.

        Returns
        -------
        sofa : :class:`sofar.Sofa`
            SOFA object representing the Dataset data.
        """

    @abstractmethod
    def _source_filename(self, **dataset_kwargs) -> str:
        """Construct the ingest-ready filename with extension for the dataset.

        Override in subclass.

        This name is canonical: ``_get``
        treats the existence of that path as proof that download *and*
        processing are already done (if so it skips both and ingests the file
        directly). The name therefore must match the file that actually
        ends up on disk after ``_download`` + ``_process``: i.e. the *processed*
        file (merged/extracted), which is not necessarily the raw download.

        Parameters
        ----------
        **dataset_kwargs : dict
            Dataset-specific parameters used to construct the filename.

        Returns
        -------
        str
            The ingest-ready filename including extension (e.g., "A1.h5",
            "FABIAN_HRIR_measured_HATO_0.sofa").
        """

    def _output_path(self, output_dir: Path, source_filename : str, output_format: str) -> Path | None:
        """Return the canonical Path where a file-based output would be written.

        Returns None for formats ('pyfar', 'numpy', 'raw'). Constructs the Path based on filename,
        directory target and output format.

        Parameters
        ----------
        output_dir : Path
            The output directory. Either cache_dir/output, export_dir, or export_dir/raw.
        source_filename : str
            The name of the ingestible file. Constructed with _source_filename
        output_format : str
            One of 'pyfar', 'numpy', 'hdf5', 'sofa', 'raw'.

        Returns
        -------
        Path or None
            Canonical output path, or None for in-memory formats.
        """
        match output_format:
            case "numpy" | "pyfar" | "raw":
                return None
            case "sofa":
                suff = ".sofa"
            case "hdf5":
                suff = ".h5"
        return (output_dir / source_filename.stem).with_suffix(suff)

    def _export_raw(self, provider_artifact: Path, export_dir: Path) -> Path:
        """Export raw provider artifact to export directory.

        For file artifacts, copies the file with its actual name.
        For directory artifacts, copies all contents to the output base directory.
        Raises ValueError if provider_artifact is neither a file nor a directory.

        Parameters
        ----------
        provider_artifact : Path
            Path to the downloaded artifact (file or directory).
        export_dir : Path
            Target export directory.

        Returns
        -------
        Path
            Path to the exported file or directory.
        """
        output_base = export_dir / self.name.upper() / "raw"
        output_base.mkdir(exist_ok=True, parents=True)

        if provider_artifact.is_file():
            output_path = output_base / provider_artifact.name
            if not output_path.exists():
                shutil.copy2(provider_artifact, output_path)
            return output_path
        elif provider_artifact.is_dir():
            shutil.copytree(provider_artifact, output_base, dirs_exist_ok=True)
            return output_base
        else:
            raise ValueError(
                f"Provider artifact must be a file or directory, but {self.name} returned: {provider_artifact}"
            )

    def process(self, provider_artifact: Path, ingest_path: Path, **dataset_kwargs) -> Path:
        """Post-process downloaded file if needed.

        This method wraps _process to enforce ingest_dir existence for all subclasses.

        Parameters
        ----------
        provider artifact : Path
            Path to the freshly downloaded file (or download directory).
        ingest_path : :class:`pathlib.Path`
            Path to the ingestible file in the ingest directory.
        **dataset_kwargs : dict
            Dataset-specific parameters.

        Returns
        -------
        ingest_path : Path
            The processed, ingest-ready file at ``ingest_path``.
        """
        ingest_path.parent.mkdir(parents=True, exist_ok=True)
        return self._process(provider_artifact, ingest_path, **dataset_kwargs)

    def _process(self, provider_artifact: Path, ingest_path: Path, **dataset_kwargs) -> Path:
        """Post-process downloaded file if needed.

        Override in subclass to extract, merge, or otherwise transform the downloaded data. Write
        the processed, ingest-ready file to ``ingest_path`` and return it.

        The default implementation promotes the provider file to the ingest
        stage. If the provider path is a file and differs from the ingest path,
        it creates a hard link (or falls back to a copy) so the ingest file
        exists.

        Parameters
        ----------
        provider artifact : Path
            Path to the freshly downloaded file (or download directory).
        ingest_path : :class:`pathlib.Path`
            Path to the ingestible file in the ingest directory.
        **dataset_kwargs : dict
            Dataset-specific parameters.

        Returns
        -------
        ingest_path : Path
            The processed, ingest-ready file at ``ingest_path``.
        """
        if provider_artifact.is_file():
            try:
                os.link(provider_artifact, ingest_path)
            except OSError:
                shutil.copy2(provider_artifact, ingest_path)
            return ingest_path
        else:
            raise NotImplementedError(
                "BaseDataset._process can only handle single files."
                "Override _process with special implementation in subclass."
            )

    def _to_output(self, sofa: sf.Sofa, output_format: str, ingest_path: Path, output_path: Path | None) -> dict | Path:
        """Convert sofar.Sofa to the requested output format.

        Parameters
        ----------
        sofa : :class:`sofar.Sofa`
            SOFA object to convert.
        output_format : str
            One of "pyfar", "numpy", "hdf5", "sofa".
        ingest_path : :class:`pathlib.Path`
            Path to the ingestible file. We also pass to allow for file-based export mechanics that
            avoid loading into memory.
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
            return self._to_sofa(sofa, ingest_path, output_path)
        elif output_format == "hdf5":
            return self._to_hdf5(sofa, ingest_path, output_path)
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

    def _to_sofa(self, sofa: sf.Sofa, ingest_path: Path, output_path: Path) -> Path:
        """Write sofar.Sofa to file and return Path.

        Parameters
        ----------
        sofa : :class:`sofar.Sofa`
            SOFA object to write.
        ingest_path : :class:`pathlib.Path`
            Path to the ingestible file.
        output_path : :class:`pathlib.Path`
            Path where the .sofa file should be written.

        Returns
        -------
        :class:`pathlib.Path`
            Path to the written SOFA file.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write_sofa(output_path, sofa)
        return output_path

    def _to_hdf5(self, sofa: sf.Sofa, ingest_path: Path, output_path: Path) -> Path:
        """Convert sofar.Sofa to HDF5 file and return Path.

        Parameters
        ----------
        sofa : :class:`sofar.Sofa`
            SOFA object to convert.
        ingest_path : :class:`pathlib.Path`
            Path to the ingestible file.
        output_path : :class:`pathlib.Path`
            Path where the .h5 file should be written.

        Returns
        -------
        :class:`pathlib.Path`
            Path to the written HDF5 file.
        """
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
