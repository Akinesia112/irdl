"""Implements download and post-processing based on pooch."""

import pooch as po
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TimeRemainingColumn, TransferSpeedColumn

from irdl.repositories import doi_to_repository

#: The cache directory for storage of the temporary downloads. Defaults to the user cache directory.
CACHE_DIR = po.os_cache("irdl")


class RichProgressBar:
    """Wrap rich.progress.Progress to satisfy the pooch progress bar interface.

    Pooch expects an object with a ``total`` attribute and ``update``, ``reset``, and
    ``close`` methods. This class provides that interface backed by a Rich progress bar.
    """

    def __init__(self, description: str, preset_total: int = 0):
        self._progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
        )
        self._description = description
        self._task_id = None
        # Pooch sets self.total from the HTTP Content-Length header. If the server omits
        # that header, pooch sets it to 0. In that case, fall back to the preset value
        # from the repository API so the bar can show real progress.
        self._preset_total = preset_total
        self.total = 0

    @property
    def total(self) -> int:
        """Total download size in bytes.

        Returns
        -------
        int
            Total download size in bytes.
        """
        return self._total

    @total.setter
    def total(self, value):
        # Use the API-supplied size when the server omits Content-Length (value == 0).
        self._total = value or self._preset_total
        if self._task_id is not None:
            self._progress.update(self._task_id, total=self._total or None)

    def update(self, n: int) -> None:
        """Advance the progress bar by n bytes."""
        if self._task_id is None:
            self._progress.start()
            self._task_id = self._progress.add_task(self._description, total=self.total if self.total else None)
        self._progress.advance(self._task_id, n)

    def reset(self) -> None:
        """Reset the completed byte count to zero.

        Called by pooch before the final fill.
        """
        if self._task_id is not None:
            self._progress.reset(self._task_id, total=self.total if self.total else None)

    def close(self) -> None:
        """Fill to 100% and stop the progress display."""
        if self._task_id is not None:
            if self.total:
                self._progress.update(self._task_id, completed=self.total)
            self._progress.stop()
            self._task_id = None


def _fetch(pup: po.Pooch, fname: str) -> str:
    """Fetch a file from a pooch registry, displaying a Rich progress bar.

    Parameters
    ----------
    pup : pooch.Pooch
        The Pooch instance managing the registry.
    fname : str
        The file name to fetch (must be registered in pup).

    Returns
    -------
    full_path : str
        The absolute path to the fetched file on disk.

    """
    preset_total = getattr(pup, "file_sizes", {}).get(fname) or 0
    return pup.fetch(fname, progressbar=RichProgressBar(fname, preset_total=preset_total))


def _pooch_from_doi(doi: str, path: str = CACHE_DIR) -> po.Pooch:
    """Create a Pooch instance from a DOI.

    Parameters
    ----------
    doi : str
        The DOI of the archive.
    path : str, optional
        Path to the directory where the data should be stored. Default is CACHE_DIR.

    Returns
    -------
    pup : pooch.Pooch
        The Pooch instance.

    """
    pup = po.create(path=path, base_url=doi, retry_if_failed=2, env="IRDL_CACHE_DIR")
    repository = doi_to_repository(doi)
    repository.populate_registry(pup)
    for file in pup.registry.keys():
        pup.urls[file] = repository.download_url(file_name=file)
    # Attach file sizes from the repository API for use by the progress bar.
    if hasattr(repository, "file_size"):
        pup.file_sizes = {file: repository.file_size(file_name=file) for file in pup.registry}
    else:
        pup.file_sizes = {}
    return pup
