"""Implements download and post-processing based on pooch."""

import pooch as po

from irdl.logger import RichProgressBar, logger
from irdl.repositories import doi_to_repository

#: The cache directory for storage of the temporary downloads. Defaults to the user cache directory.
IRDL_CACHE_DIR = po.os_cache("irdl")


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
    logger.debug(f"Fetching {fname}")
    preset_total = getattr(pup, "file_sizes", {}).get(fname) or 0
    return pup.fetch(fname, progressbar=RichProgressBar(fname, preset_total=preset_total))


def _pooch_from_doi(doi: str, path: str = IRDL_CACHE_DIR) -> po.Pooch:
    """Create a Pooch instance from a DOI.

    Parameters
    ----------
    doi : str
        The DOI of the archive.
    path : str, optional
        Path to the directory where the data should be stored. Default is IRDL_CACHE_DIR.

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
