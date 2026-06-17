"""Shared dataset support for SOFA-backed datasets from the SONICOM ecosystem."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from pathlib import Path

from irdl.base import SofaBaseDataset
from irdl.downloader import _fetch, _pooch_from_static_registry


class SonicomBaseDataset(SofaBaseDataset):
    """Abstract base class for SOFA-backed datasets served from SONICOM.

    SONICOM is modeled as a non-canonical Provider that serves SOFA artifacts
    through direct fetch specifications rather than DOI resolution. Concrete
    subclasses provide the small registry and URL maps needed to download the
    requested file.
    """

    @abstractmethod
    def _source_filename(self, **dataset_kwargs) -> str:
        """Return the ingest-ready SOFA filename for the requested SONICOM artifact."""

    @abstractmethod
    def _sonicom_registry(self, **dataset_kwargs) -> Mapping[str, str | None]:
        """Return pooch registry entries for SONICOM downloads."""

    @abstractmethod
    def _sonicom_urls(self, **dataset_kwargs) -> Mapping[str, str]:
        """Return direct download URLs for SONICOM downloads."""

    def _provider_artifact_format(self, provider: str, **_dataset_kwargs) -> str:
        """Return the Provider-side artifact Data Format for SONICOM datasets.

        SONICOM-backed Provider artifacts are expected to be SOFA files.
        """
        if provider != "sonicom":
            msg = f"Unknown provider {provider!r} for {self.name.upper()}"
            raise ValueError(msg)
        return "sofa"

    def _download(self, provider_dir: Path, provider: str, **dataset_kwargs) -> Path:
        """Download the selected SONICOM SOFA file from a direct fetch spec.

        Parameters
        ----------
        provider_dir : Path
            Provider directory for the SONICOM cache stage.
        provider : str
            Provider name. Must be ``"sonicom"``.
        **dataset_kwargs : dict
            Dataset-specific parameters used to choose one SONICOM file.

        Returns
        -------
        Path
            Path to the downloaded SOFA file inside the SONICOM Provider cache.
        """
        if provider != "sonicom":
            msg = f"Unknown provider {provider!r} for {self.name.upper()}"
            raise ValueError(msg)
        source_filename = self._source_filename(**dataset_kwargs)
        self.logger.info("provider=%r artifact=%r -> download to provider cache", provider, source_filename)
        pup = _pooch_from_static_registry(
            path=provider_dir,
            registry=self._sonicom_registry(**dataset_kwargs),
            urls=self._sonicom_urls(**dataset_kwargs),
        )
        return Path(_fetch(pup, source_filename))
