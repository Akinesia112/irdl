"""Impulse response datasets in SOFA format.

This module provides the FABIAN Dataset implementation using the BaseDataset
architecture, along with legacy helper functions for backwards compatibility.
"""

from pathlib import Path
from zipfile import ZipFile

import pooch as po
import sofar as sf

from irdl.base import BaseDataset
from irdl.downloader import CACHE_DIR, _fetch, _pooch_from_doi


class FabianDataset(BaseDataset):
    """Implement FABIAN HRTF Database Dataset.

    Attributes
    ----------
    name : str
        Dataset name ("fabian").
    doi : str
        Digital Object Identifier ("10.14279/depositonce-5718.5").
    """

    name = "fabian"
    doi = "10.14279/depositonce-5718.5"

    def validate_params(self, **dataset_kwargs) -> None:
        """Validate FABIAN-specific parameters.

        Parameters
        ----------
        **dataset_kwargs : dict
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
        **kwargs : dict
            Expected keys: kind, hato.

        Returns
        -------
        str
            File name in format "FABIAN_HRIR_{kind}_HATO_{hato}.sofa".
        """
        kind = kwargs["kind"]
        hato = kwargs["hato"]
        return f"FABIAN_HRIR_{kind}_HATO_{hato}.sofa"

    def download(self, target_path: Path, **kwargs) -> Path:
        """Download FABIAN dataset and extract the requested SOFA file.

        Downloads the ZIP archive if needed, then extracts the specific SOFA file
        based on kind and hato parameters.

        Parameters
        ----------
        target_path : :class:`pathlib.Path`
            Target path where the SOFA file should be extracted.
        **kwargs : dict
            Expected keys: kind, hato.

        Returns
        -------
        :class:`pathlib.Path`
            Path to the extracted SOFA file.
        """
        base_dir = target_path.parent

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
                if name.endswith(target_path.name):
                    zf.getinfo(name).filename = Path(name).name
                    logger.info(f"Extracting {name} to {target_path}")
                    zf.extract(name, path=base_dir)

        return target_path

    @classmethod
    def get(
        cls,
        kind: str = "measured",
        hato: int = 0,
        cache_dir: str | Path = CACHE_DIR,
        export_dir: str | Path | None = None,
        output_format: str = "pyfar",
    ):
        """
        kind : str, optional
            Type of HRTF to download. Either 'measured' or 'modeled'. Default is 'measured'.
        hato : int, optional
            Head-above-torso-rotation of HRTFs in degrees.
            One of: 0, 10, 20, 30, 40, 50, 310, 320, 330, 340, 350. Default is 0.
        """  # noqa: D205, D403
        return cls()._get(
            kind=kind,
            hato=hato,
            cache_dir=cache_dir,
            export_dir=export_dir,
            output_format=output_format,
        )

    def ingest(self, file_path: Path) -> sf.Sofa:
        """Load SOFA file into sofar.Sofa object.

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
