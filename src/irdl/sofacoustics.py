"""SOFACoustics-hosted datasets with direct static SOFA downloads."""

from pathlib import Path

from irdl.base import DatasetCategory
from irdl.downloader import _fetch, _pooch_from_static_registry
from irdl.logging import logger
from irdl.sofa import SofaBaseDataset
from irdl.utils import load_hash_registry


def get_sofacoustics_hash(path_key: str) -> str:
    """Return one verified hash entry from the SOFACoustics provider registry."""
    registry = load_hash_registry("sofacoustics")
    try:
        return registry[path_key]
    except KeyError as exc:
        msg = f"Missing SOFACoustics hash registry entry for '{path_key}'"
        raise ValueError(msg) from exc


class SofacousticsBaseDataset(SofaBaseDataset):
    """Base class for datasets hosted as static files on sofacoustics.org."""

    dataset_slug: str

    def _provider_filename(self, **dataset_kwargs) -> str:
        """Return the provider file name for the requested dataset variant."""
        return self._source_filename(**dataset_kwargs)

    def _provider_url(self, filename: str) -> str:
        """Return the direct download URL for a provider file."""
        return f"https://sofacoustics.org/data/database/{self.dataset_slug}/{filename}"

    def _download(self, provider_dir: Path, **dataset_kwargs) -> Path:
        """Download one static provider file using the checked-in hash registry."""
        filename = self._provider_filename(**dataset_kwargs)
        path_key = f"{self.dataset_slug}/{filename}"
        known_hash = get_sofacoustics_hash(path_key)
        pup = _pooch_from_static_registry(
            path=provider_dir,
            registry={filename: known_hash},
            urls={filename: self._provider_url(filename)},
        )
        logger.info(f"Downloading {self.name.upper()} file {filename}")
        _fetch(pup, filename)
        return provider_dir / filename


class HutubsDataset(SofacousticsBaseDataset):
    """Download the HUTUBS HRTF database from SOFAcoustics."""

    name = "hutubs"
    doi = "10.14279/depositonce-8487"
    dataset_slug = "hutubs"
    _category = DatasetCategory.HEAD_RELATED_IMPULSE_RESPONSES

    @classmethod
    def get(
        cls,
        subject: int = 1,
        kind: str = "measured",
        cache_dir: str | Path | None = None,
        export_dir: str | Path | None = None,
        output_format: str = "pyfar",
    ) -> dict | Path | None:
        """
        subject : int, optional
            Subject identifier. Must be an integer in the range 1 to 96. Default is 1.
        kind : str, optional
            HUTUBS HRIR variant. Either 'measured' or 'simulated'. Default is 'measured'.

        Returns
        -------
        dict or Path
            For 'pyfar' / 'numpy': dict of in-memory objects.
            For 'sofa' / 'hdf5' / 'raw': Path to file on disk.
        """  # noqa: D205, D403
        return cls()._get(
            subject=subject,
            kind=kind,
            cache_dir=cache_dir,
            export_dir=export_dir,
            output_format=output_format,
        )

    def _validate_params(self, **dataset_kwargs) -> None:
        """Validate HUTUBS-specific parameters."""
        subject = dataset_kwargs["subject"]
        kind = dataset_kwargs["kind"]

        if not isinstance(subject, int):
            msg = "subject must be an integer in the range 1 to 96"
            raise TypeError(msg)
        if subject not in range(1, 97):
            msg = "subject must be an integer in the range 1 to 96"
            raise ValueError(msg)
        if kind not in ["measured", "simulated"]:
            msg = "kind must be either 'measured' or 'simulated'"
            raise ValueError(msg)

    def _source_filename(self, **dataset_kwargs) -> str:
        """Construct the ingest-ready SOFA file name."""
        return f"pp{dataset_kwargs['subject']}_HRIRs_{dataset_kwargs['kind']}.sofa"
