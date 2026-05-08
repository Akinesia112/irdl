"""Impulse response datasets in SOFA format."""

from pathlib import Path
from zipfile import ZipFile

import h5py as h5
import numpy as np
import pooch as po
import pyfar as pf

from irdl.downloader import CACHE_DIR, _fetch, _pooch_from_doi
from irdl.utils import _fits_in_memory, _move_to_export_dir


def _load_sofa(
    file: Path,
):
    """Load raw arrays from a SOFA file.

    Parameters
    ----------
    file : :class:`pathlib.Path` or :class:`str`
        Path to the SOFA file.

    Returns
    -------
    data : :class:`dict`
        Dictionary with the following keys:

        - ``'impulse_response'`` : :class:`numpy.ndarray` — Impulse response data.
        - ``'source_coordinates'`` : :class:`numpy.ndarray` — Source positions as cartesian
          coordinates.
        - ``'receiver_coordinates'`` : :class:`numpy.ndarray` — Receiver positions as cartesian
          coordinates.
        - ``'sampling_rate'`` : :class:`float` — Sampling rate in Hz.

    """
    pyfar_obj = pf.io.read_sofa(file)

    return {
        "impulse_response": pyfar_obj[0].time,
        "source_coordinates": pyfar_obj[1].cartesian,
        # fabian need the squeeze, maybe other datasets dont?
        "receiver_coordinates": np.squeeze(pyfar_obj[2].cartesian, axis=1),
        "sampling_rate": pyfar_obj[0].sampling_rate,
    }


def _sofa_to_pyfar(
    file: Path,
):
    """Load data from a SOFA file and return as dictionary of pyfar objects.

    Parameters
    ----------
    file : :class:`pathlib.Path` or :class:`str`
        Path to the SOFA file.

    Returns
    -------
    data : :class:`dict`
        Dictionary with the following keys:

        - ``'impulse_response'`` : :class:`pyfar.Signal` — Impulse response data.
        - ``'source_coordinates'`` : :class:`pyfar.Coordinates` — Source positions.
        - ``'receiver_coordinates'`` : :class:`pyfar.Coordinates` — Receiver positions.

    """
    return dict(
        zip(
            ("impulse_response", "source_coordinates", "receiver_coordinates"),
            pf.io.read_sofa(file),
            strict=True,
        )
    )


def _sofa_to_h5(
    file: Path,
):
    """Convert a SOFA file to HDF5 format and return the path.

    Parameters
    ----------
    file : :class:`pathlib.Path` or :class:`str`
        Path to the SOFA file.

    Returns
    -------
    h5_path : :class:`pathlib.Path`
        Path to the converted HDF5 file.

    """
    h5_path = Path(file).with_suffix(".h5")
    data = _load_sofa(file)
    with h5.File(h5_path, "w") as f:
        data_group = f.create_group("data")
        data_group.create_dataset("impulse_response", data=data["impulse_response"])
        location_group = data_group.create_group("location")
        location_group.create_dataset("source", data=data["source_coordinates"])
        location_group.create_dataset("receiver", data=data["receiver_coordinates"])
        metadata_group = f.create_group("metadata")
        metadata_group.create_dataset("sampling_rate", data=data["sampling_rate"])

    return h5_path


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
        by the environment variable ``IRDL_DATA_DIR`` when set. Defaults to the
        user cache directory.
    export_dir : :class:`str` or :class:`pathlib.Path` or None
        Directory to move the output file to after processing. When ``None``
        (default) the output file stays in ``cache_dir``.
    output_format : :class:`str`
        Output format of the returned data.
        Either ``'pyfar'`` (default), ``'hdf5'``, ``'sofa'`` or ``'numpy'``.

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
    assert kind in ["measured", "modeled"], "kind must be either 'measured' or 'modeled'"
    assert hato in [0, 10, 20, 30, 40, 50, 310, 320, 330, 340, 350], (
        "hato must be one of [0, 10, 20, 30, 40, 50, 310, 320, 330, 340, 350]"
    )
    assert output_format in ["pyfar", "hdf5", "numpy", "sofa"], "unknown output format"

    doi = "10.14279/depositonce-5718.5"
    zipfile_name = "FABIAN_HRTF_DATABASE_v4.zip"
    cache_dir = Path(cache_dir) / "FABIAN"

    # paths
    base_name = f"FABIAN_HRIR_{kind}_HATO_{hato}"
    sofa_cache = cache_dir / f"{base_name}.sofa"
    h5_cache = cache_dir / f"{base_name}.h5"
    sofa_export = Path(export_dir) / f"{base_name}.sofa" if export_dir else None
    h5_export = Path(export_dir) / f"{base_name}.h5" if export_dir else None

    # use file if it already exists, preferring export
    sofa_path = sofa_export if sofa_export is not None and sofa_export.exists() else sofa_cache
    h5_path = h5_export if h5_export is not None and h5_export.exists() else h5_cache

    # extract sofa from zip if we need it and don't have it yet
    sofa_exists = sofa_cache.exists() or (sofa_export is not None and sofa_export.exists())
    h5_exists = h5_cache.exists() or (h5_export is not None and h5_export.exists())
    needs_sofa = (output_format == "hdf5" and not h5_exists) or (output_format != "hdf5" and not sofa_exists)
    if needs_sofa and not sofa_exists:
        pup = _pooch_from_doi(doi, path=cache_dir)
        _fetch(pup, zipfile_name)
        logger = po.get_logger()
        with ZipFile(cache_dir / zipfile_name, "r") as zf:
            for name in zf.namelist():
                if name.endswith(sofa_cache.name):
                    zf.getinfo(name).filename = Path(name).name
                    logger.info(f"Extracting {name} to {sofa_cache.parent / Path(name).name}")
                    zf.extract(name, path=cache_dir)

    # check if the file can be loaded into memory if not, fall back to hdf5
    if output_format in ["pyfar", "numpy"] and not _fits_in_memory(sofa_path):
        output_format = "hdf5"

    match output_format:
        case "hdf5":
            if not h5_path.exists():
                h5_path = _sofa_to_h5(sofa_path)
            return _move_to_export_dir(h5_path, export_dir)

        case "sofa":
            return _move_to_export_dir(sofa_path, export_dir)

        case "pyfar":
            # Export original file, so .sofa
            if export_dir is not None and not sofa_export.exists():
                _move_to_export_dir(sofa_cache, export_dir)
                return _sofa_to_pyfar(sofa_export)
            return _sofa_to_pyfar(sofa_path)

        case "numpy":
            # Export original file, so .sofa
            if export_dir is not None and not sofa_export.exists():
                _move_to_export_dir(sofa_cache, export_dir)
                return _load_sofa(sofa_export)
            return _load_sofa(sofa_path)
