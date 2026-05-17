"""Impulse response datasets in SOFA format.

This module provides the FABIAN Dataset implementation using the BaseDataset
architecture, along with legacy helper functions for backwards compatibility.
"""

from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pooch as po

from irdl.base import CACHE_DIR, BaseDataset
from irdl.downloader import _fetch, _pooch_from_doi


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

    def validate_params(self, dataset_kwargs: dict) -> None:
        """Validate FABIAN-specific parameters.

        Parameters
        ----------
        dataset_kwargs : :class:`dict`
            Parameters to validate. Expected keys: kind, hato.
        """
        kind = dataset_kwargs["kind"]
        hato = dataset_kwargs["hato"]

        if kind not in ["measured", "modeled"]:
            raise ValueError("kind must be either 'measured' or 'modeled'")
        if hato not in [0, 10, 20, 30, 40, 50, 310, 320, 330, 340, 350]:
            raise ValueError("hato must be one of [0, 10, 20, 30, 40, 50, 310, 320, 330, 340, 350]")

    def _construct_file_name(self, **kwargs) -> str:
        """Construct file name based on kind and hato parameters.

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

    def download(self, **kwargs) -> Path:
        """Download FABIAN ZIP archive.

        Parameters
        ----------
        **kwargs : :class:`dict`
            Expected keys: kind, hato, cache_dir.

        Returns
        -------
        :class:`pathlib.Path`
            Path to the downloaded ZIP file.
        """
        cache_dir = Path(kwargs.get("cache_dir", CACHE_DIR)) / "FABIAN"
        cache_dir.mkdir(parents=True, exist_ok=True)

        zipfile_name = "FABIAN_HRTF_DATABASE_v4.zip"
        zip_path = cache_dir / zipfile_name

        # Download ZIP archive if not exists
        if not zip_path.exists():
            pup = _pooch_from_doi(self.doi, path=cache_dir)
            _fetch(pup, zipfile_name)

        return zip_path

    def _process(self, file_path: Path, **kwargs) -> Path:
        """Extract SOFA file from FABIAN ZIP archive.

        Parameters
        ----------
        file_path : :class:`pathlib.Path`
            Path to the ZIP file.
        **kwargs : :class:`dict`
            Expected keys: kind, hato.

        Returns
        -------
        :class:`pathlib.Path`
            Path to the extracted SOFA file.
        """
        kind = kwargs["kind"]
        hato = kwargs["hato"]
        cache_dir = file_path.parent
        base_name = f"FABIAN_HRIR_{kind}_HATO_{hato}"
        sofa_path = cache_dir / f"{base_name}.sofa"

        # Check if SOFA file already exists
        if sofa_path.exists():
            return sofa_path

        # Extract SOFA file from ZIP
        logger = po.get_logger()
        with ZipFile(file_path, "r") as zf:
            for name in zf.namelist():
                if name.endswith(sofa_path.name):
                    zf.getinfo(name).filename = Path(name).name
                    logger.info(f"Extracting {name} to {sofa_path}")
                    zf.extract(name, path=cache_dir)

        return sofa_path

    @classmethod
    def get(
        cls,
        kind: str = "measured",
        hato: int = 0,
        cache_dir: str = CACHE_DIR,
        export_dir: str = None,
        output_format: str = "pyfar",
    ):
        """Kind : str
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

    def ingest(self, file_path: Path) -> Any:
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
        import sofar as sf

        return sf.read_sofa(str(file_path))
