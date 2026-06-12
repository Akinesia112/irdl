"""Impulse response datasets in SOFA format.

This module provides the SofaBaseDataset class (for datasets whose provider
format is already SOFA or archived SOFA) and the FABIAN Dataset implementation,
along with legacy helper functions for backwards compatibility.
"""

import os
import shutil
from pathlib import Path
from zipfile import ZipFile

import sofar as sf

from irdl.base import BaseDataset, DatasetCategory
from irdl.downloader import _fetch, _pooch_from_doi
from irdl.logging import logger


class SofaBaseDataset(BaseDataset):
    """Base class for datasets whose ingest-ready format is already SOFA.

    The primary distinction is that ``output_format='sofa'`` can either directly copy or link the
    ingest-ready file, avoiding having to write the sofa file in memory.
    """

    def _to_sofa(self, sofa: sf.Sofa, ingest_path: Path, output_path: Path) -> Path:  # noqa: ARG002
        """Copy sofar.Sofa file from ingest_dir and return Path.

        Parameters
        ----------
        sofa : :class:`sofar.Sofa`
            SOFA object to write.
        ingest_path : :class:`pathlib.Path`
            Path to the ingestible file.
        output_path : :class:`pathlib.Path`
            Path where the .sofa file should be written to.

        Returns
        -------
        :class:`pathlib.Path`
            Path to the written SOFA file.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.parent.parent == ingest_path.parent.parent:
            try:
                logger.debug(f"Linking {ingest_path} to {output_path}.")
                os.link(ingest_path, output_path)
            except OSError as e:
                logger.debug(f"Linking failed: {e!r}")
            else:
                return output_path
        logger.debug(f"Copying {ingest_path} to {output_path}.")
        shutil.copy2(ingest_path, output_path)
        return output_path

    def _ingest(self, ingest_path: Path) -> sf.Sofa:
        """Load SOFA file into sofar.Sofa object.

        Parameters
        ----------
        ingest_path : :class:`pathlib.Path`
            Path to the SOFA file in the ingest directory.

        Returns
        -------
        :class:`sofar.Sofa`
            SOFA object containing the dataset data.
        """
        return sf.read_sofa(ingest_path)


class FabianDataset(SofaBaseDataset):
    """Download and extract the FABIAN HRTF Database from DepositOnce.

    Attributes
    ----------
    name : str
        Dataset name ("fabian").
    doi : str
        Digital Object Identifier ("10.14279/depositonce-5718.5").
    """

    name = "fabian"
    doi = "10.14279/depositonce-5718.5"
    _category = DatasetCategory.HEAD_RELATED_IMPULSE_RESPONSES

    @classmethod
    def get(
        cls,
        kind: str = "measured",
        hato: int = 0,
        cache_dir: str | Path | None = None,
        export_dir: str | Path | None = None,
        output_format: str = "pyfar",
    ) -> dict | Path | None:
        """
        kind : str, optional
            Type of HRTF to download. Either 'measured' or 'modeled'. Default is 'measured'.
        hato : int, optional
            Head-above-torso-rotation of HRTFs in degrees.
            One of: 0, 10, 20, 30, 40, 50, 310, 320, 330, 340, 350. Default is 0.

        Returns
        -------
        dict or Path
            For 'pyfar' / 'numpy': dict of in-memory objects.
            For 'sofa' / 'hdf5' / 'raw': Path to file on disk.
        """  # noqa: D205, D403
        return cls()._get(
            kind=kind,
            hato=hato,
            cache_dir=cache_dir,
            export_dir=export_dir,
            output_format=output_format,
        )

    def _validate_params(self, **dataset_kwargs) -> None:
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
            msg = "kind must be either 'measured' or 'modeled'"
            raise ValueError(msg)
        if hato not in [0, 10, 20, 30, 40, 50, 310, 320, 330, 340, 350]:
            msg = "hato must be one of [0, 10, 20, 30, 40, 50, 310, 320, 330, 340, 350]"
            raise ValueError(msg)

    def _source_filename(self, **dataset_kwargs) -> str:
        """Construct the ingest-ready (SOFA) filename.

        Parameters
        ----------
        **dataset_kwargs : dict
            Expected keys: kind, hato.

        Returns
        -------
        str
            File name in format "FABIAN_HRIR_{kind}_HATO_{hato}.sofa".
        """
        return f"FABIAN_HRIR_{dataset_kwargs['kind']}_HATO_{dataset_kwargs['hato']}.sofa"

    def _download(self, provider_dir: Path, **dataset_kwargs) -> Path:  # noqa: ARG002
        """Download FABIAN ZIP archive to the provider directory.

        Only downloads the archive if it is not already cached in the provider
        directory. Returns the ZIP path so that ``_process`` can extract the
        requested SOFA file into the ingest directory.

        Parameters
        ----------
        provider_dir : :class:`pathlib.Path`
            Provider directory (e.g., ``cache/FABIAN/provider/``).
        **dataset_kwargs : dict
            Expected keys: kind, hato (unused here because the ZIP contains all
            variants).

        Returns
        -------
        :class:`pathlib.Path`
            Path to the downloaded ZIP archive.
        """
        zipfile_name = "FABIAN_HRTF_DATABASE_v4.zip"
        zip_path = provider_dir / zipfile_name
        if zip_path.exists():
            logger.info(f"FABIAN ZIP archive already cached at {zip_path}, skipping download")
        else:
            pup = _pooch_from_doi(self.doi, path=provider_dir)
            _fetch(pup, zipfile_name)
        return zip_path

    def _process(self, provider_artifact: Path, ingest_path: Path) -> Path:
        """Extract the requested SOFA file from the ZIP into the ingest directory.

        Parameters
        ----------
        provider_artifact : :class:`pathlib.Path`
            Path to the ZIP archive in the provider directory.
        ingest_path : :class:`pathlib.Path`
            Path to the SOFA file in the ingest directory.

        Returns
        -------
        :class:`pathlib.Path`
            Path to the extracted SOFA file in the ingest directory.
        """
        with ZipFile(provider_artifact, "r") as zf:
            for name in zf.namelist():
                if name.endswith(ingest_path.name):
                    # Flatten the extraction (strip any nested ZIP directory)
                    zf.getinfo(name).filename = Path(name).name
                    logger.info(f"Extracting {name} to {ingest_path.parent}")
                    zf.extract(name, path=ingest_path.parent)
                    return ingest_path

            msg = (
                f"No entry matching '{ingest_path.name}' found in archive {provider_artifact}. "
                "Check zf.namelist() for available entries."
            )
            raise FileNotFoundError(msg)
