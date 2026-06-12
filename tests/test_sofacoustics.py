"""Tests for SOFACoustics-hosted datasets."""

from pathlib import Path

import pytest

from irdl.sofacoustics import HutubsDataset
from irdl.sofacoustics_registry import SOFACOUSTICS_HASHES


class TestHutubsDataset:
    """Tests for HUTUBS dataset parameter and provider mapping behavior."""

    def test_validate_accepts_documented_subject_range(self):
        """Verify subject values 1 and 96 are accepted."""
        dataset = HutubsDataset()

        dataset._validate_params(subject=1, kind="measured", output_format="raw")
        dataset._validate_params(subject=96, kind="simulated", output_format="sofa")

    @pytest.mark.parametrize(
        ("subject", "error_type"),
        [(0, ValueError), (97, ValueError), (-1, ValueError), (1.5, TypeError), ("1", TypeError)],
    )
    def test_validate_rejects_invalid_subject(self, subject, error_type):
        """Verify HUTUBS rejects subject values outside 1..96 or non-integers."""
        dataset = HutubsDataset()

        with pytest.raises(error_type, match="subject must be an integer in the range 1 to 96"):
            dataset._validate_params(subject=subject, kind="measured", output_format="raw")

    @pytest.mark.parametrize("kind", ["modeled", "raw", "Measured"])
    def test_validate_rejects_invalid_kind(self, kind):
        """Verify HUTUBS rejects unsupported kind values."""
        dataset = HutubsDataset()

        with pytest.raises(ValueError, match="kind must be either 'measured' or 'simulated'"):
            dataset._validate_params(subject=1, kind=kind, output_format="raw")

    def test_source_filename_matches_provider_naming(self):
        """Verify ingest-ready file name follows SOFACoustics naming."""
        dataset = HutubsDataset()

        assert dataset._source_filename(subject=12, kind="simulated") == "pp12_HRIRs_simulated.sofa"
        assert dataset._provider_filename(subject=12, kind="simulated") == "pp12_HRIRs_simulated.sofa"

    def test_provider_url_uses_dataset_slug(self):
        """Verify direct provider URLs are built from the dataset slug."""
        dataset = HutubsDataset()

        assert (
            dataset._provider_url("pp12_HRIRs_simulated.sofa")
            == "https://sofacoustics.org/data/database/hutubs/pp12_HRIRs_simulated.sofa"
        )

    def test_hash_registry_covers_supported_hutubs_hrirs(self):
        """Verify the provider-wide registry covers all supported HUTUBS HRIR SOFA files."""
        hutubs_hashes = SOFACOUSTICS_HASHES["hutubs"]
        expected_supported_files = 96 * 2

        assert len(hutubs_hashes) == expected_supported_files
        assert hutubs_hashes["pp1_HRIRs_measured.sofa"].startswith("sha256:")
        assert hutubs_hashes["pp96_HRIRs_simulated.sofa"].startswith("sha256:")

    def test_download_uses_static_registry_entry(self, monkeypatch, tmp_path):
        """Verify download constructs a single-file static registry for the provider file."""
        dataset = HutubsDataset()
        captured: dict[str, object] = {}

        class DummyPooch:
            pass

        def fake_pooch_from_static_registry(path, registry, urls):
            captured["path"] = path
            captured["registry"] = registry
            captured["urls"] = urls
            return DummyPooch()

        def fake_fetch(pup, fname):
            captured["pup"] = pup
            captured["fname"] = fname
            target = Path(captured["path"]) / fname
            target.write_text("placeholder")
            return str(target)

        monkeypatch.setattr("irdl.sofacoustics._pooch_from_static_registry", fake_pooch_from_static_registry)
        monkeypatch.setattr("irdl.sofacoustics._fetch", fake_fetch)

        result = dataset._download(tmp_path, subject=3, kind="measured")

        assert result == tmp_path / "pp3_HRIRs_measured.sofa"
        assert captured["registry"] == {
            "pp3_HRIRs_measured.sofa": SOFACOUSTICS_HASHES["hutubs"]["pp3_HRIRs_measured.sofa"]
        }
        assert captured["urls"] == {
            "pp3_HRIRs_measured.sofa": "https://sofacoustics.org/data/database/hutubs/pp3_HRIRs_measured.sofa"
        }
        assert captured["fname"] == "pp3_HRIRs_measured.sofa"
