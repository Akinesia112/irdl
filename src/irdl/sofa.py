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

    def validate_params(self, kwargs: dict) -> None:
        """Validate FABIAN-specific parameters.

        Parameters
        ----------
        kwargs : :class:`dict`
            Parameters to validate. Expected keys: kind, hato.
        """
        kind = kwargs.get("kind", "measured")
        hato = kwargs.get("hato", 0)

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
        kind = kwargs.get("kind", "measured")
        hato = kwargs.get("hato", 0)
        return f"FABIAN_HRIR_{kind}_HATO_{hato}.sofa"

    def download(self, **kwargs) -> Path:
        """Download FABIAN ZIP archive and extract SOFA file.

        Parameters
        ----------
        **kwargs : :class:`dict`
            Expected keys: kind, hato, cache_dir.

        Returns
        -------
        :class:`pathlib.Path`
            Path to the extracted SOFA file.
        """
        kind = kwargs.get("kind", "measured")
        hato = kwargs.get("hato", 0)
        cache_dir = Path(kwargs.get("cache_dir", CACHE_DIR)) / "FABIAN"
        cache_dir.mkdir(parents=True, exist_ok=True)

        zipfile_name = "FABIAN_HRTF_DATABASE_v4.zip"
        base_name = f"FABIAN_HRIR_{kind}_HATO_{hato}"
        sofa_cache = cache_dir / f"{base_name}.sofa"

        # Check if SOFA file already exists
        if sofa_cache.exists():
            return sofa_cache

        # Download ZIP archive
        pup = _pooch_from_doi(self.doi, path=cache_dir)
        _fetch(pup, zipfile_name)

        # Extract SOFA file from ZIP
        logger = po.get_logger()
        with ZipFile(cache_dir / zipfile_name, "r") as zf:
            for name in zf.namelist():
                if name.endswith(sofa_cache.name):
                    zf.getinfo(name).filename = Path(name).name
                    logger.info(f"Extracting {name} to {sofa_cache}")
                    zf.extract(name, path=cache_dir)

        return sofa_cache

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


# Singleton instance
fabian_dataset = FabianDataset()


# Backwards-compatible public API
def get_fabian(
    kind: str = "measured",
    hato: int = 0,
    cache_dir: str = CACHE_DIR,
    export_dir: str = None,
    output_format: str = "pyfar",
):
    """Download and extract the FABIAN HRTF Database v4 from DepositOnce.

    DOI: `10.14279/depositonce-5718.5 <https://doi.org/10.14279/depositonce-5718.5>`_

    Parameters
    ----------
    kind : :class:`str`
        Type of HRTF to download. Either ``'measured'`` or ``'modeled'``.
    hato : :class:`int`
        Head-above-torso-rotation of HRTFs in degrees.
        Either 0, 10, 20, 30, 40, 50, 310, 320, 330, 340 or 350.
    cache_dir : :class:`str` or :class:`pathlib.Path`
        Directory used to store raw downloads and intermediate files. Overridden
        by the environment variable ``IRDL_CACHE_DIR`` when set. Defaults to the
        user cache directory.
    export_dir : :class:`str` or :class:`pathlib.Path` or :class:`None`
        Directory to move the output file to after processing. When ``None``
        (default) the output file stays in ``cache_dir``.
    output_format : :class:`str`
        Output format of the returned data.
        Either ``'pyfar'`` (default), ``'hdf5'``, ``'numpy'`` or ``'sofa'``.

    Returns
    -------
    data : :class:`dict` or :class:`pathlib.Path`
        Returned data depends on ``output_format``:

        - ``'pyfar'`` : :class:`dict` with keys ``'impulse_response'`` (:class:`pyfar.Signal`),
          ``'source_coordinates'`` (:class:`pyfar.Coordinates`), and
          ``'receiver_coordinates'`` (:class:`pyfar.Coordinates`).
        - ``'hdf5'`` : :class:`pathlib.Path` to the HDF5 file containing the data.
        - ``'sofa'`` : :class:`pathlib.Path` to the SOFA file.
        - ``'numpy'`` : :class:`dict` with keys ``'impulse_response'`` (:class:`numpy.ndarray`),
          ``'source_coordinates'`` (:class:`numpy.ndarray`),
          ``'receiver_coordinates'`` (:class:`numpy.ndarray`), and
          ``'sampling_rate'`` (:class:`float`).
    """
    return fabian_dataset.get(
        kind=kind, hato=hato, cache_dir=cache_dir, export_dir=export_dir, output_format=output_format
    )
