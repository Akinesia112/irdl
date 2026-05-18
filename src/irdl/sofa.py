"""Impulse response datasets in SOFA format.

This module provides the FABIAN Dataset implementation using the BaseDataset
architecture, along with legacy helper functions for backwards compatibility.
"""

from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pooch as po
import sofa as sf

from irdl.base import BaseDataset
from irdl.downloader import CACHE_DIR, _fetch, _pooch_from_doi


class FabianDataset(BaseDataset):
    """Implement FABIAN HRTF Database Dataset.

    Attributes
    ----------
    name : :class:`str`
        Dataset name ("fabian").
    doi : :class:`str`
        Digital Object Identifier ("10.14279/depositonce-5718.5").
    """

    name = "fabian"
    doi = "10.14279/depositonce-5718.5"

    def validate_params(self, **dataset_kwargs) -> None:
        """Validate FABIAN-specific parameters.

        Parameters
        ----------
        **dataset_kwargs
            Parameters to validate. Expected keys: kind, hato.

        Raises
        ------
        ValueError
            If kind or hato is out of range.
        """
        kind = dataset_kwargs["kind"]
        hato = dataset_kwargs["hato"]

        if kind not in ["measured", "modeled"]:
            raise ValueError("kind must be either 'measured' or 'modeled'")
        if hato not in [0, 10, 20, 30, 40, 50, 310, 320, 330, 340, 350]:
            raise ValueError("hato must be one of [0, 10, 20, 30, 40, 50, 310, 320, 330, 340, 350]")

    def _source_filename(self, **kwargs) -> str:
        """Construct the raw input filename with extension.

        For FABIAN, this is the SOFA file that gets extracted from the ZIP archive.

        Parameters
        ----------
        **kwargs : :class:`dict`
            Expected keys: kind, hato.

        Returns
        -------
        :class:`str`
            File name in format "FABIAN_HRIR_{kind}_HATO_{hato}.sofa".
        """
        kind = kwargs["kind"]
        hato = kwargs["hato"]
        return f"FABIAN_HRIR_{kind}_HATO_{hato}.sofa"

    def _get_file(self, cache_dir: Path, export_dir: Path | None, **kwargs) -> Path:
        """Return the path to a FABIAN SOFA file, extracting from ZIP if needed.

        Parameters
        ----------
        cache_dir : :class:`pathlib.Path`
            Base cache directory.
        export_dir : :class:`pathlib.Path` or :class:`None`
            Optional export directory; takes priority over cache_dir.
        **kwargs : :class:`dict`
            Expected keys: kind, hato.

        Returns
        -------
        :class:`pathlib.Path`
            Path to the SOFA file, ready for ingest().
        """
        # Get the expected SOFA file path
        sofa_path = self._input_path(cache_dir, export_dir, **kwargs)

        # Check if SOFA file already exists
        if sofa_path.exists():
            return sofa_path

        # Need to download and extract from ZIP
        base_dir = (export_dir if export_dir is not None else cache_dir) / self.name.upper()
        base_dir.mkdir(parents=True, exist_ok=True)

        # Download ZIP if needed
        zipfile_name = "FABIAN_HRTF_DATABASE_v4.zip"
        zip_path = base_dir / zipfile_name
        if not zip_path.exists():
            pup = _pooch_from_doi(self.doi, path=base_dir)
            _fetch(pup, zipfile_name)

        # Extract SOFA file from ZIP
        logger = po.get_logger()
        with ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                if name.endswith(sofa_path.name):
                    zf.getinfo(name).filename = Path(name).name
                    logger.info(f"Extracting {name} to {sofa_path}")
                    zf.extract(name, path=sofa_path.parent)

        return sofa_path

    @classmethod
    def get(
        cls,
        kind: str = "measured",
        hato: int = 0,
        cache_dir: str | Path = CACHE_DIR,
        export_dir: str | Path | None = None,
        output_format: str = "pyfar",
    ):
        """Download FABIAN dataset.

DOI: 10.14279/depositonce-5718.5

Parameters
----------
cache_dir : str
    Cache directory for downloads. Default: user cache directory.
export_dir : str, optional
    Directory for final output. Default: None (stays in cache_dir).
output_format : str
    Output format: 'pyfar', 'numpy', 'hdf5', 'sofa', or 'raw'.

kind : str
    Type of HRTF to download. Either 'measured' or 'modeled'.
hato : int
    Head-above-torso-rotation of HRTFs in degrees.
    One of: 0, 10, 20, 30, 40, 50, 310, 320, 330, 340, 350.
"""
        instance = cls()
        return instance._get(
            kind=kind,
            hato=hato,
            cache_dir=cache_dir,
            export_dir=export_dir,
            output_format=output_format,
        )

    def ingest(self, file_path: Path) -> sf.Sofa:
        """Load SOFA file into :class:`sofar.Sofa` object.

        Parameters
        ----------
        file_path : :class:`pathlib.Path`
            Path to the SOFA file.

        Returns
        -------
        :class:`sofar.Sofa`
            SOFA object containing the dataset data.
        """
        return sf.read_sofa(str(file_path))
