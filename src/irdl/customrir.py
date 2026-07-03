"""Datasets with custom provider formats.

This module provides Dataset implementations whose provider data requires
custom handling to build the internal SOFA representation
(e.g. WAV files, proprietary archives).

Currently this module hosts:

- MYRiAD: A Multi-Array Room Acoustic Database (KU Leuven).

"""

import json
import tempfile
from pathlib import Path
from typing import ClassVar
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pyfar as pf
import sofar as sf

from irdl.base import BaseDataset, DatasetCategory
from irdl.downloader import _fetch, _pooch_from_doi
from irdl.logging import logger


class MyriadDataset(BaseDataset):
    """Download the MYRiAD database (RIR subset) from Zenodo.

    Attributes
    ----------
    name : str
        Dataset name ("myriad").
    doi : str
        Digital Object Identifier ("10.5281/zenodo.7389996").
    """

    name = "myriad"
    doi = "10.5281/zenodo.7389996"
    _category = DatasetCategory.ROOM_IMPULSE_RESPONSES

    # Selectable array-group labels and their corresponding microphone labels
    _ARRAY_GROUPS: ClassVar[dict[str, list[str]]] = {
        "dummy_head": ["DHL", "DHR"],
        "bte-pieces": ["BTELF", "BTELB", "BTERF", "BTERB"],
        "external-microphones": ["XM1", "XM2", "XM3", "XM4", "XM5"],
        "circular-microphone-array": ["CMA10_-90", "CMA10_0", "CMA10_90",
            "CMA10_180", "CMA20_-135", "CMA20_-90", "CMA20_-45", "CMA20_0",
            "CMA20_45", "CMA20_90", "CMA20_135", "CMA20_180",],
    }

    # WORK ON VARIABLES FROM HERE ON 

    # TODO: confirm against the archive. The econ RIRs are 3.0 s @ 44.1 kHz.
    _SAMPLING_RATE = 44100
    _N_SAMPLES = 132300

    #: Loudspeaker (emitter) labels per room, in canonical order.
    _SPEAKERS: ClassVar[dict[str, list[str]]] = {
        "SAL": ["S0_1", "S0_2", "S-30_1", "S30_1", "S-45_2",
            "S45_2", "S-60_1", "S60_1", "S-90_1", "S90_1",],
        "AIL": ["SL1", "SL2", "SL3", "SL4", "SL5", "SL6",
            "SL7", "SL8", "SU1", "SU2", "SU3", "SU4", "SU5",
            "SU6", "SU7", "SU8", "SU9", "SU10", "SU11", "SU12",
            "ST1", "ST2", "ST3", "ST4",],
    }

    #: Human-readable receiver hardware per group (for ReceiverDescriptions).
    _GROUP_HARDWARE: ClassVar[dict[str, str]] = {
        "dummy_head": "Neumann KU 100 in-ear microphone",
        "bte-pieces": "Cochlear behind-the-ear microphone",
        "external-microphones": "External microphone",
        "circular-microphone-array": "Circular microphone array (DPA 4060 / AKG CK32)",
    }

    # TODO: replace with measured room volumes [m^3] (paper gives floor plans
    # but not heights). Used for SOFA RoomVolume metadata.
    _ROOM_VOLUME: ClassVar[dict[str, float | None]] = {"SAL": None, "AIL": None}

    @classmethod
    def get(
        cls,
        room: str = "SAL",
        array: str = "all",
        config: str = "P1",
        convention: str = "SingleRoomMIMOSRIR",
        cache_dir: str | Path | None = None,
        export_dir: str | Path | None = None,
        output_format: str = "pyfar",
    ) -> dict | Path | None:
        """
        Room : str
            Room to load. Either ``'SAL'`` or ``'AIL'``. Default is ``'SAL'``.
        array : str
            Microphone array group(s) to load. One of ``'dummy_head'``,
            ``'bte-pieces'``, ``'external-microphones'``, ``'circular-microphone-array'``,
            a comma-separated combination of these (e.g. ``'dummy_head,bte-pieces'``),
            or ``'all'`` for every group available in the room.
            A list of group names is also accepted via the Python API.
            The circular array is only available in ``'AIL'``. Default is ``'all'``.
        config : str
            Microphone configuration placement in the AIL. Either ``'P1'`` or ``'P2'``.
            Ignored for the SAL. Default is ``'P1'``.
        convention : str or None
            SOFA convention of the output. Either ``'SingleRoomMIMOSRIR'`` or
            ``'MultiSpeakerBRIR'``. ``'MultiSpeakerBRIR'`` is only valid when
            ``array`` selects exactly the ``'dummy_head'`` group. Default is
            ``'SingleRoomMIMOSRIR'``.

        Returns
        -------
        dict or Path
            For 'pyfar' / 'numpy': dict of in-memory objects.
            For 'sofa' / 'hdf5' / 'raw': Path to file on disk.
        """
        return cls()._get(
            room=room,
            array=array,
            config=config,
            convention=convention,
            cache_dir=cache_dir,
            export_dir=export_dir,
            output_format=output_format,
        )

    def _validate_params(self, **dataset_kwargs) -> None:
        """Validate MYRiAD-specific parameters.

        Parameters
        ----------
        **dataset_kwargs : dict
            Must contain ``'room'`` (one of ``'SAL'``, ``'AIL'``), ``'array'`` (one
            or more valid group names or ``'all'``), ``'config'`` (one of ``'P1'``,
            ``'P2'``), and ``'convention'`` (one of ``'SingleRoomMIMOSRIR'``,
            ``'MultiSpeakerBRIR'``). ``output_format`` is also passed but unused here.

        Raises
        ------
        ValueError
            If any parameter is out of range, the circular array is requested for
            ``'SAL'``, or ``'MultiSpeakerBRIR'`` is combined with more than the
            ``'dummy_head'`` group.
        """
        parameters = self._resolve(**dataset_kwargs)

        if parameters["room"] not in ("SAL", "AIL"):
            raise ValueError("room must be one of ['SAL', 'AIL']")

        for group in parameters["groups"]:
            if group not in self._ARRAY_GROUPS:
                raise ValueError(f"array group(s) {group!r} must be one of {list(self._ARRAY_GROUPS)} or 'all'")
            if parameters["room"] == "SAL" and group == "circular-microphone-array":
                raise ValueError(f"array group {group!r} is not available in the SAL")

        if parameters["room"] == "AIL" and sel["config"] not in ("P1", "P2"):
            raise ValueError("config must be one of ['P1', 'P2'] for the AIL")

        if parameters["convention"] not in ("SingleRoomMIMOSRIR", "MultiSpeakerBRIR"):
            raise ValueError("convention must be one of ['SingleRoomMIMOSRIR', 'MultiSpeakerBRIR']")
        
        if parameters["convention"] == "MultiSpeakerBRIR" and set(sel["groups"]) != {"dummy_head"}:
            raise ValueError("convention 'MultiSpeakerBRIR' is only valid for array='dummy_head'")

    def _source_filename(self, **dataset_kwargs) -> str:
        """Build the ingest filename encoding the full selection.

        The name has the form
        ``MYRIAD_<room>[_<config>]_<groups>_<convention>.zip``, where ``<config>``
        is included only for the AIL and ``<groups>`` is the selected array groups
        as short tokens. ``<convention>`` is part of the name because the ingest
        archive embeds the convention in its metadata sidecar, so each convention
        must map to a distinct cache file.

        Returns
        -------
        str
            Filename for the ingest-stage archive.
        """
        # Short tokens used to build a unique, filesystem-friendly cache filename.
        _GROUP_TOKEN = {
            "dummy_head": "dh",
            "bte-pieces": "bte",
            "external-microphones": "xm",
            "circular-microphone-array": "cma",
        }
        parameters = self._resolve(**dataset_kwargs)

        tokens = "-".join(_GROUP_TOKEN[g] for g in parameters["groups"])
        parts = ["MYRIAD", parameters["room"]]
        if parameters["room"] == "AIL":
            parts.append(parameters["config"])
        parts += [tokens, parameters["convention"]]
        return "_".join(parts)

    def _download(self, provider_dir: Path, **dataset_kwargs) -> Path:
        """Download MYRiAD ZIP archive to the provider directory.

        Only downloads the archive if it is not already cached. Returns the 
        ZIP path so that ``_process`` can extract and process the requested files.

        Parameters
        ----------
        provider_dir : :class:`pathlib.Path`
            Provider directory (e.g., ``cache/MYRiAD/provider/``).
        **dataset_kwargs : dict
            
        Returns
        -------
        :class:`pathlib.Path`
            Path to the downloaded ZIP archive.
        """
        zipfile_name = "MYRiAD_V2_econ.zip"
        zip_path = provider_dir / zipfile_name
        if zip_path.exists():
            logger.info(f"MYRiAD ZIP archive already cached at {zip_path}, skipping download")
        else:
            logger.info("Downloading MYRiAD (econ RIR archive)")
            pup = _pooch_from_doi(self.doi, path=provider_dir)
            _fetch(pup, zipfile_name)
        return zip_path

    def _process(self, provider_artifact: Path, ingest_path: Path, **dataset_kwargs) -> Path:
        """Extract only the selected RIR WAVs and coordinate CSV into the ingest directory.

        Performs no signal processing: selects the WAVs and the room coordinate CSV for 
        the requested selection and extracts them, preserving the archive tree, into 
        ``ingest_path``. All SOFA construction is deferred to :meth:`_ingest`.

        Parameters
        ----------
        provider_artifact : :class:`pathlib.Path`
            Path to the downloaded MYRiAD ZIP archive.
        ingest_path : :class:`pathlib.Path`
            Target directory for the extracted files.

        Returns
        -------
        :class:`pathlib.Path`
            Path to the ingest directory (``ingest_path``).
        """
        parameters = self._resolve(**dataset_kwargs)
        ingest_path.mkdir(parents=True, exist_ok=True)

        # build strings for each file that needs to be extracted
        files = []
        for speaker in parameters["speakers"]:
            base = f"MYRiAD_V2_econ/audio/{parameters['room']}/{speaker}"
            if parameters["room"] == "AIL":
                base = f"{base}/{parameters['config']}"
            files += [f"{base}/{mic}_RIR.wav" for mic in parameters["mics"]]
        files.append(f"MYRiAD_V2_econ/coord/{parameters['room']}.csv")

        # extract files while preserving the archive tree
        with ZipFile(provider_artifact, "r") as zf:
            for member in files:
                logger.debug(f"Extracting {member}")
                zf.extract(member, path=ingest_path)  

        return ingest_path








    def _ingest(self, ingest_path: Path) -> sf.Sofa:  # noqa: PLR0915 - single cohesive ingest pipeline
        """Build the internal SOFA from the ingest archive written by :meth:`_process`.

        Reads the ``meta.json`` sidecar for the resolved selection, stacks the
        selected loudspeaker/microphone WAVs into a ``Data.IR`` array of shape
        ``(M, R, N, E)``, attaches the geometry from the room coordinate CSV, and
        layers on the descriptive metadata. The loudspeakers are the *emitters*
        (E) and the selected microphones are the *receivers* (R) — the inverse of
        the moving-source ISTA datasets, where source positions index M.

        Parameters
        ----------
        ingest_path : :class:`pathlib.Path`
            Path to the ingest-stage ZIP archive.

        Returns
        -------
        :class:`sofar.Sofa`
            The fully populated internal SOFA representation.
        """
        with ZipFile(ingest_path, "r") as zf:
            meta = json.loads(zf.read("meta.json"))
            room = meta["room"]
            convention = meta["convention"]
            speakers = meta["speakers"]
            mics = meta["mics"]

            # --- stack the impulse responses, shape (M=1, R, N, E) -----------
            ir = np.zeros((1, len(mics), self._N_SAMPLES, len(speakers)), dtype=np.float32)
            with tempfile.TemporaryDirectory() as tmp:
                for ei, speaker in enumerate(speakers):
                    for ri, mic in enumerate(mics):
                        member = f"audio/{speaker}/{mic}_RIR.wav"
                        wav_path = Path(zf.extract(member, path=tmp))
                        # pyfar returns time data of shape (channels, samples)
                        time = pf.io.read_audio(wav_path).time[0]
                        # TODO: handle length mismatches instead of truncating.
                        ir[0, ri, : time.shape[0], ei] = time[: self._N_SAMPLES]

                csv_path = Path(zf.extract("coord.csv", path=tmp))
                # --- coordinates: CSV columns are Label, x, y, z -------------
                rows = np.atleast_2d(np.genfromtxt(csv_path, delimiter=",", dtype=str, skip_header=1))

        coords = {row[0].strip(): row[1:4].astype(float) for row in rows}
        # TODO: verify CSV labels match the speaker/mic labels exactly.
        source = np.array([coords[s] for s in speakers])  # (E, C)
        receiver = np.array([coords[mic] for mic in mics])  # (R, C)

        m, r, _n, e = ir.shape
        listener_pos = receiver.mean(axis=0)  # array centre, (C,)
        source_pos = source.mean(axis=0)  # source frame origin, (C,)

        # --- build the SOFA holding the raw data and geometry ----------------
        sofa = sf.Sofa(convention)
        sofa.Data_IR = ir  # (M=1, R, N, E)
        sofa.Data_SamplingRate = float(self._SAMPLING_RATE)
        sofa.Data_Delay = np.zeros((m, r, 1))  # TODO: verify required dims for E>1
        sofa.ListenerPosition = listener_pos[np.newaxis, :]  # (M=1, C)
        sofa.ReceiverPosition = receiver.reshape(r, 3, 1)  # (R, C, I)
        sofa.SourcePosition = source_pos[np.newaxis, :]  # (M=1, C)
        sofa.EmitterPosition = source.reshape(e, 3, 1)  # (E, C, I)

        label_to_group = {mic: group for group, labels in self._ARRAY_GROUPS.items() for mic in labels}
        descriptions = [self._GROUP_HARDWARE[label_to_group[mic]] for mic in mics]

        # --- global metadata -------------------------------------------------
        sofa.GLOBAL_Title = "MYRiAD"
        sofa.GLOBAL_DatabaseName = "MYRiAD"
        sofa.GLOBAL_Organization = "KU Leuven, ESAT-STADIUS"
        sofa.GLOBAL_AuthorContact = "randall.ali@kuleuven.be"  # TODO: confirm
        sofa.GLOBAL_References = f"https://doi.org/{self.doi}"
        sofa.GLOBAL_RoomType = "reverberant"
        sofa.GLOBAL_Comment = f"MYRiAD room {room}; emitters=loudspeakers, receivers=selected microphones."

        # --- coordinate types/units and receiver descriptions ---------------
        sofa.ListenerPosition_Type = "cartesian"
        sofa.ListenerPosition_Units = "metre"
        sofa.ReceiverPosition_Type = "cartesian"
        sofa.ReceiverPosition_Units = "metre"
        sofa.ReceiverDescriptions = np.array(descriptions)  # (R, S)
        sofa.SourcePosition_Type = "cartesian"
        sofa.SourcePosition_Units = "metre"
        sofa.EmitterPosition_Type = "cartesian"
        sofa.EmitterPosition_Units = "metre"

        if self._ROOM_VOLUME[room] is not None:
            sofa.RoomVolume = self._ROOM_VOLUME[room]

        # --- convention-specific extras --------------------------------------
        if convention == "MultiSpeakerBRIR":
            # Binaural head orientation; +x look direction, +z up. TODO: refine
            # from the actual dummy-head orientation in the room.
            sofa.ListenerView = np.tile([1.0, 0.0, 0.0], (m, 1))
            sofa.ListenerView_Type = "cartesian"
            sofa.ListenerUp = np.tile([0.0, 0.0, 1.0], (m, 1))

        return sofa

    @classmethod
    def _resolve(cls, *, room, array, config=None, convention=None, **_ignored) -> dict:
        """Resolve the raw get() parameters into a concrete selection.

        Parses the array selection into a deduplicated, canonical-order group list
        and bundles it with the room, config, convention, and the resolved speaker
        and microphone labels. This method performs no validation: unknown group
        names are kept at the end of ``'groups'`` so that ``_validate_params`` can
        reject them, and ``config`` is passed through untouched.

        Parameters
        ----------
        room : str
            Room being loaded. Used to look up the speakers and to exclude the
            circular array from ``'all'`` for room SAL.
        array : str or list of str
            Requested group(s): ``'all'``, a comma-separated string, or a list of
            the array group names.
        config : str or None
            Microphone configuration placement in the AIL (``'P1'`` or ``'P2'``).
            Ignored for the SAL.
        convention : str or None
            Requested SOFA convention, or None to default to
            ``'SingleRoomMIMOSRIR'``.

        Returns
        -------
        dict
            Resolved selection with keys ``'room'``, ``'config'``, ``'convention'``,
            ``'groups'`` (requested groups in canonical order, unknowns kept at the
            end), ``'speakers'`` (emitter labels) and ``'mics'`` (receiver labels
            for the known groups).
        """
        if array == "all":
            groups = [g for g in cls._ARRAY_GROUPS if room != "SAL" or g != "circular-microphone-array"]
        elif isinstance(array, str):
            groups = [g.strip() for g in array.split(",")]
        else:
            groups = list(array)

        # canonical order for known groups, unknowns kept at the end for validation
        seen = set(groups)
        groups = [g for g in cls._ARRAY_GROUPS if g in seen] + [g for g in groups if g not in cls._ARRAY_GROUPS]

        return {
            "room": room,
            "config": config,
            "convention": convention or "SingleRoomMIMOSRIR",
            "groups": groups,
            #check if we need this
            "speakers": cls._SPEAKERS[room],
            "mics": [mic for g in groups if g in cls._ARRAY_GROUPS for mic in cls._ARRAY_GROUPS[g]],
        }