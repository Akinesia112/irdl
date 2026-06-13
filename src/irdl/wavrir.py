"""Datasets shipped as plain WAV room impulse responses inside a single archive.

Unlike the HDF5-based ISTA datasets or the SOFA-native datasets, the providers
covered here distribute one (potentially large) ZIP archive that contains the
RIRs as individual ``.wav`` files plus separate coordinate tables. The shared
pattern is therefore:

1. download the archive once (``_download``),
2. extract the WAV/coordinate members needed for the requested selection and
   build an initial SOFA file holding only the raw data (``_process``),
3. read that file back and enrich it with descriptive metadata (``_ingest``).

Currently this module hosts:

- MYRiAD: Multi-arraY Room Acoustic Database (KU Leuven).

.. note::
    This is a first draft. Several details are marked ``TODO`` and must be
    confirmed against the actual ``MYRiAD_V2_econ.zip`` layout before relying on
    the output. The :class:`~irdl.base.BaseDataset` flow runs ``sofa.verify()``
    and prints actionable diagnostics; use them to iterate on the convention
    metadata.
"""

from pathlib import Path
from typing import ClassVar
from zipfile import ZipFile

import numpy as np
import pyfar as pf
import sofar as sf

from irdl.base import BaseDataset, DatasetCategory
from irdl.downloader import _fetch, _pooch_from_doi
from irdl.logging import logger


class MyriadDataset(BaseDataset):
    """Download the MYRiAD database (RIR subset) from Zenodo.

    MYRiAD (Multi-arraY Room Acoustic Database) contains room impulse responses
    measured in two reverberant rooms (SAL, T20 ~= 2.1 s; AIL, T20 ~= 0.5 s)
    with several microphone configurations. This loader uses the compact
    ``MYRiAD_V2_econ.zip`` archive, which holds the *computed RIRs only*.

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

    # archive layout
    _ZIP = "MYRiAD_V2_econ.zip"
    _ROOT = "MYRiAD_V2_econ"

    # TODO: confirm against the archive. The econ RIRs are 3.0 s @ 44.1 kHz.
    _SAMPLING_RATE = 44100
    _N_SAMPLES = 132300

    _VALID_CONVENTIONS = ("SingleRoomMIMOSRIR", "MultiSpeakerBRIR")

    #: Loudspeaker (emitter) labels per room, in canonical order.
    _SPEAKERS: ClassVar[dict[str, list[str]]] = {
        "SAL": [
            "S0_1", "S0_2", "S-30_1", "S30_1", "S-45_2",
            "S45_2", "S-60_1", "S60_1", "S-90_1", "S90_1",
        ],
        "AIL": [
            "SL1", "SL2", "SL3", "SL4", "SL5", "SL6", "SL7", "SL8",
            "SU1", "SU2", "SU3", "SU4", "SU5", "SU6", "SU7", "SU8",
            "SU9", "SU10", "SU11", "SU12",
            "ST1", "ST2", "ST3", "ST4",
        ],
    }

    #: Microphone (receiver) labels per array group, in canonical order.
    _ARRAY_GROUPS: ClassVar[dict[str, list[str]]] = {
        "dummy_head": ["DHL", "DHR"],
        "bte-pieces": ["BTELF", "BTELB", "BTERF", "BTERB"],
        "external-microphones": ["XM1", "XM2", "XM3", "XM4", "XM5"],
        "circular-microphone-array": [
            "CMA10_-90", "CMA10_0", "CMA10_90", "CMA10_180",
            "CMA20_-135", "CMA20_-90", "CMA20_-45", "CMA20_0",
            "CMA20_45", "CMA20_90", "CMA20_135", "CMA20_180",
        ],
    }

    #: Short tokens used to build a unique, filesystem-friendly cache filename.
    _GROUP_TOKEN: ClassVar[dict[str, str]] = {
        "dummy_head": "dh",
        "bte-pieces": "bte",
        "external-microphones": "xm",
        "circular-microphone-array": "cma",
    }

    #: Human-readable receiver hardware per group (for ReceiverDescriptions).
    _GROUP_HARDWARE: ClassVar[dict[str, str]] = {
        "dummy_head": "Neumann KU 100 in-ear microphone",
        "bte-pieces": "Cochlear behind-the-ear microphone",
        "external-microphones": "External microphone",
        "circular-microphone-array": "Circular microphone array (DPA 4060 / AKG CK32)",
    }

    #: The circular array only exists in the AIL.
    _SAL_GROUPS = frozenset({"dummy_head", "bte-pieces", "external-microphones"})

    # TODO: replace with measured room volumes [m^3] (paper gives floor plans
    # but not heights). Used for SOFA RoomVolume metadata.
    _ROOM_VOLUME: ClassVar[dict[str, float | None]] = {"SAL": None, "AIL": None}

    @classmethod
    def get(
        cls,
        room: str = "SAL",
        array: str = "all",
        config: str = "P1",
        convention: str | None = None,
        cache_dir: str | Path | None = None,
        export_dir: str | Path | None = None,
        output_format: str = "pyfar",
    ) -> dict | Path | None:
        """
        room : str
            Room to load. One of 'SAL' or 'AIL'.
        array : str
            Microphone array group(s) to load. One of 'dummy_head',
            'bte-pieces', 'external-microphones', 'circular-microphone-array',
            a comma-separated combination of these (e.g.
            'dummy_head,bte-pieces'), or 'all' for every group available in the
            room. A list of group names is also accepted via the Python API.
            The circular array is only available in 'AIL'.
        config : str
            Microphone configuration placement in the AIL. One of 'P1' or 'P2'.
            Ignored for the SAL.
        convention : str or None
            SOFA convention of the output. One of 'SingleRoomMIMOSRIR',
            'MultiSpeakerBRIR', or None (defaults to 'SingleRoomMIMOSRIR').
            'MultiSpeakerBRIR' is only valid when ``array`` selects exactly the
            'dummy_head' group.

        Returns
        -------
        dict or Path
            For 'pyfar' / 'numpy': dict of in-memory objects.
            For 'sofa' / 'hdf5' / 'raw': Path to file on disk.
        """  # noqa: D205, D403
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
        """Validate parameters and resolve the selection onto the instance.

        The resolved room, config, convention, loudspeaker list and microphone
        list are cached on ``self`` so that ``_source_filename``, ``_process``
        and ``_ingest`` can use them directly. ``_validate_params`` always runs
        before those methods in the :class:`~irdl.base.BaseDataset` flow.
        """
        room = dataset_kwargs["room"]
        array = dataset_kwargs["array"]
        config = dataset_kwargs.get("config")
        convention = dataset_kwargs.get("convention")

        if room not in ("SAL", "AIL"):
            msg = "room must be one of ['SAL', 'AIL']"
            raise ValueError(msg)

        if room == "AIL":
            allowed = list(self._ARRAY_GROUPS)
        else:
            allowed = [g for g in self._ARRAY_GROUPS if g in self._SAL_GROUPS]
        if isinstance(array, str) and array == "all":
            groups = allowed
        else:
            requested = [g.strip() for g in array.split(",")] if isinstance(array, str) else list(array)
            for group in requested:
                if group not in self._ARRAY_GROUPS:
                    msg = f"array group {group!r} must be one of {list(self._ARRAY_GROUPS)} or 'all'"
                    raise ValueError(msg)
                if room == "SAL" and group not in self._SAL_GROUPS:
                    msg = f"array group {group!r} is not available in the SAL"
                    raise ValueError(msg)
            groups = [g for g in allowed if g in set(requested)]

        if room == "AIL" and config not in ("P1", "P2"):
            msg = "config must be one of ['P1', 'P2'] for the AIL"
            raise ValueError(msg)

        if convention is not None and convention not in self._VALID_CONVENTIONS:
            msg = f"convention must be None or one of {list(self._VALID_CONVENTIONS)}"
            raise ValueError(msg)
        if convention == "MultiSpeakerBRIR" and set(groups) != {"dummy_head"}:
            msg = "convention 'MultiSpeakerBRIR' is only valid for array='dummy_head'"
            raise ValueError(msg)

        # cache resolved selection for the remaining stages
        self._room = room
        self._config = config
        self._convention = convention or "SingleRoomMIMOSRIR"
        self._groups = groups
        self._speakers = self._SPEAKERS[room]
        self._mics = [mic for group in groups for mic in self._ARRAY_GROUPS[group]]

    def _source_filename(self, **dataset_kwargs) -> str:  # noqa: ARG002
        """Build a unique ingest filename encoding the full selection."""
        tokens = "-".join(self._GROUP_TOKEN[g] for g in self._groups)
        parts = ["MYRIAD", self._room]
        if self._room == "AIL":
            parts.append(self._config)
        parts += [tokens, self._convention]
        return "_".join(parts) + ".sofa"

    def _download(self, provider_dir: Path, **dataset_kwargs) -> Path:  # noqa: ARG002
        """Download the compact MYRiAD RIR archive from Zenodo."""
        logger.info("Downloading MYRiAD (econ RIR archive)")
        pup = _pooch_from_doi(self.doi, path=provider_dir)
        return Path(_fetch(pup, self._ZIP))

    def _process(self, provider_artifact: Path, ingest_path: Path, **dataset_kwargs) -> Path:  # noqa: ARG002
        """Extract the selected RIRs and build the initial SOFA with raw data.

        Only the data arrays and geometry are written here; descriptive metadata
        is layered on later in :meth:`_ingest`. Both supported conventions use
        ``Data.IR`` with dimensions ``(M, R, N, E)``; the loudspeakers are the
        *emitters* (E) and the selected microphones are the *receivers* (R).
        This is the inverse of the moving-source ISTA datasets, where the source
        positions index M.
        """
        room, config, convention = self._room, self._config, self._convention
        speakers, mics = self._speakers, self._mics

        work_dir = ingest_path.parent / "_extracted"
        work_dir.mkdir(parents=True, exist_ok=True)

        # --- stack the impulse responses, shape (M=1, R, N, E) ---------------
        ir = np.zeros((1, len(mics), self._N_SAMPLES, len(speakers)), dtype=np.float32)
        with ZipFile(provider_artifact, "r") as zf:
            for ei, speaker in enumerate(speakers):
                base = f"{self._ROOT}/audio/{room}/{speaker}"
                if room == "AIL":
                    base = f"{base}/{config}"
                for ri, mic in enumerate(mics):
                    member = f"{base}/{mic}_RIR.wav"
                    logger.debug(f"Extracting {member}")
                    wav_path = Path(zf.extract(member, path=work_dir))
                    # pyfar returns time data of shape (channels, samples)
                    time = pf.io.read_audio(wav_path).time[0]
                    # TODO: handle length mismatches instead of truncating.
                    ir[0, ri, : time.shape[0], ei] = time[: self._N_SAMPLES]

            csv_path = Path(zf.extract(f"{self._ROOT}/coord/{room}.csv", path=work_dir))

        # --- coordinates: CSV columns are Label, x, y, z ---------------------
        rows = np.atleast_2d(np.genfromtxt(csv_path, delimiter=",", dtype=str, skip_header=1))
        coords = {row[0].strip(): row[1:4].astype(float) for row in rows}
        # TODO: verify CSV labels match the speaker/mic labels exactly.
        source = np.array([coords[s] for s in speakers])  # (E, C)
        receiver = np.array([coords[mic] for mic in mics])  # (R, C)

        m, r, _n, e = ir.shape
        listener_pos = receiver.mean(axis=0)  # array centre, (C,)
        source_pos = source.mean(axis=0)  # source frame origin, (C,)

        # --- build the initial SOFA holding only the raw data ----------------
        sofa = sf.Sofa(convention)
        sofa.Data_IR = ir  # (M=1, R, N, E)
        sofa.Data_SamplingRate = float(self._SAMPLING_RATE)
        sofa.Data_Delay = np.zeros((m, r, 1))  # TODO: verify required dims for E>1
        sofa.ListenerPosition = listener_pos[np.newaxis, :]  # (M=1, C)
        sofa.ReceiverPosition = receiver.reshape(r, 3, 1)  # (R, C, I)
        sofa.SourcePosition = source_pos[np.newaxis, :]  # (M=1, C)
        sofa.EmitterPosition = source.reshape(e, 3, 1)  # (E, C, I)

        sf.write_sofa(ingest_path, sofa)
        return ingest_path

    def _ingest(self, ingest_path: Path) -> sf.Sofa:
        """Read the raw-data SOFA and enrich it with descriptive metadata."""
        room, convention, mics = self._room, self._convention, self._mics
        sofa = sf.read_sofa(ingest_path)
        m = sofa.Data_IR.shape[0]

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
