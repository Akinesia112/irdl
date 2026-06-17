"""Tests for shared SONICOM dataset support."""

from pathlib import Path

import pytest

from irdl.akt import HutubsDataset
from irdl.sonicom import SonicomBaseDataset


class ConcreteSonicomDataset(SonicomBaseDataset):
    """Concrete SONICOM dataset for testing shared behavior."""

    name = "dummy-sonicom"
    doi = "10.0000/dummy-canonical"
    canonical_provider = "depositonce"
    providers = ("depositonce", "sonicom")

    def _validate_params(self, **dataset_kwargs) -> None:
        if "sofa_file" not in dataset_kwargs:
            raise ValueError("missing sofa_file")

    def _source_filename(self, **dataset_kwargs) -> str:
        return dataset_kwargs["sofa_file"]

    def _sonicom_registry(self, **dataset_kwargs):
        return {dataset_kwargs["sofa_file"]: None}

    def _sonicom_urls(self, **dataset_kwargs):
        sofa_file = dataset_kwargs["sofa_file"]
        return {sofa_file: f"https://sonicom.example/{sofa_file}"}


class TestSonicomBaseDataset:
    """Tests for SONICOM shared dataset behavior."""

    def test_base_class_remains_abstract(self):
        """Verify SonicomBaseDataset cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            SonicomBaseDataset()

    def test_download_uses_static_registry_and_urls(self, monkeypatch, tmp_path):
        """Verify SONICOM download uses direct fetch specifications instead of DOI resolution."""
        captured = {}

        class DummyPooch:
            def __init__(self):
                self.path = tmp_path

        def fake_pooch_from_static_registry(path, registry, urls):
            captured["path"] = path
            captured["registry"] = registry
            captured["urls"] = urls
            return DummyPooch()

        def fake_fetch(pup, fname):
            captured["fname"] = fname
            target = Path(pup.path) / fname
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("ok")
            return str(target)

        monkeypatch.setattr("irdl.sonicom._pooch_from_static_registry", fake_pooch_from_static_registry)
        monkeypatch.setattr("irdl.sonicom._fetch", fake_fetch)

        result = ConcreteSonicomDataset()._download(tmp_path, provider="sonicom", sofa_file="selected.sofa")

        assert result == tmp_path / "selected.sofa"
        assert captured == {
            "path": tmp_path,
            "registry": {"selected.sofa": None},
            "urls": {"selected.sofa": "https://sonicom.example/selected.sofa"},
            "fname": "selected.sofa",
        }


class TestHutubsSonicomProvider:
    """Tests for HUTUBS using SONICOM as an additional Provider."""

    def test_provider_artifact_format_differs_by_provider(self):
        """Verify HUTUBS distinguishes canonical ZIP and SONICOM SOFA artifacts."""
        dataset = HutubsDataset()

        assert dataset._provider_artifact_format("depositonce") == "zip"
        assert dataset._provider_artifact_format("sonicom") == "sofa"

    def test_download_uses_static_sonicom_url(self, monkeypatch, tmp_path):
        """Verify HUTUBS SONICOM downloads one direct SOFA file."""
        captured = {}

        class DummyPooch:
            def __init__(self):
                self.path = tmp_path

        def fake_pooch_from_static_registry(path, registry, urls):
            captured["path"] = path
            captured["registry"] = registry
            captured["urls"] = urls
            return DummyPooch()

        def fake_fetch(pup, fname):
            captured["fname"] = fname
            target = Path(pup.path) / fname
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("ok")
            return str(target)

        monkeypatch.setattr("irdl.akt._pooch_from_static_registry", fake_pooch_from_static_registry)
        monkeypatch.setattr("irdl.akt._fetch", fake_fetch)

        result = HutubsDataset()._download(tmp_path, provider="sonicom", subject=1, kind="measured")

        assert result == tmp_path / "pp1_HRIRs_measured.sofa"
        assert captured == {
            "path": tmp_path,
            "registry": {"pp1_HRIRs_measured.sofa": None},
            "urls": {"pp1_HRIRs_measured.sofa": "https://sofacoustics.org/data/database/hutubs/pp1_HRIRs_measured.sofa"},
            "fname": "pp1_HRIRs_measured.sofa",
        }
