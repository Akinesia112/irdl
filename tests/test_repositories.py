"""Tests for repository detection and metadata parsing."""

from types import SimpleNamespace

import pytest
from requests.exceptions import Timeout

import irdl.repositories as repositories_module
from irdl.repositories import DSpaceRepository, RadarRepository, doi_to_repository


def test_dspace_initialize_requires_only_url_match(monkeypatch):
    """Verify DSpace initialization returns directly for matching URLs."""
    monkeypatch.setattr(
        DSpaceRepository,
        "_probe_repository",
        classmethod(lambda cls, archive_url: pytest.fail(f"unexpected probe for {archive_url}")),
    )

    repo = DSpaceRepository.initialize("10.14279/depositonce-123", "https://depositonce.tu-berlin.de/handle/11303/12345")

    assert isinstance(repo, DSpaceRepository)


def test_dspace_initialize_rejects_non_matching_host(monkeypatch):
    """Verify DSpace initialization rejects unrelated hosts before construction."""
    monkeypatch.setattr(
        DSpaceRepository,
        "_probe_repository",
        classmethod(lambda cls, archive_url: pytest.fail(f"unexpected probe for {archive_url}")),
    )

    repo = DSpaceRepository.initialize("10.0000/example", "https://example.org/handle/11303/12345")

    assert repo is None


def test_dspace_api_base_url_derives_api_host():
    """Verify DSpace API URLs are derived from the landing host."""
    archive_url = "https://depositonce.tu-berlin.de/handle/11303/12345"
    item_uuid = "1806cd7f-67f3-4355-b5e8-6cab392813bd"

    assert DSpaceRepository._api_base_url(archive_url) == "https://api-depositonce.tu-berlin.de"
    assert (
        DSpaceRepository._bundles_url(archive_url, item_uuid)
        == f"https://api-depositonce.tu-berlin.de/server/api/core/items/{item_uuid}/bundles"
    )


def test_radar_initialize_requires_only_url_match(monkeypatch):
    """Verify RADAR initialization returns directly for matching URLs."""
    monkeypatch.setattr(
        RadarRepository,
        "_probe_repository",
        classmethod(lambda cls, archive_url: pytest.fail(f"unexpected probe for {archive_url}")),
    )

    repo = RadarRepository.initialize("10.60887/example", "https://datathek.oeaw.ac.at/radar/en/dataset/example")

    assert isinstance(repo, RadarRepository)


def test_radar_parse_archive_checksum():
    """Verify RADAR checksum parser extracts algorithm and value."""
    html = """
    <div class="row tmd-archivechecksum">
      <div class="col-md-3">Archive checksum:</div>
      <div class="col-md-9">ee7976db28d6ce0b88cfd3ef11560d7f (MD5)</div>
    </div>
    """

    assert RadarRepository._parse_archive_checksum(html) == ("md5", "ee7976db28d6ce0b88cfd3ef11560d7f")


def test_radar_distribution_metadata_from_html():
    """Verify RADAR JSON-LD parser extracts DataDownload metadata."""
    html = """
    <script type="application/ld+json">
    {
      "@graph": [
        {
          "@id": "_:download",
          "http://schema.org/encodingFormat": "application/x-tar",
          "http://schema.org/contentUrl": "https://datathek.oeaw.ac.at/radar-backend/archives/example/versions/1/content",
          "http://schema.org/contentSize": "46069760 bytes",
          "@type": "http://schema.org/DataDownload"
        }
      ]
    }
    </script>
    """

    assert RadarRepository._distribution_metadata_from_html(html) == {
        "url": "https://datathek.oeaw.ac.at/radar-backend/archives/example/versions/1/content",
        "size": 46069760,
        "encoding_format": "application/x-tar",
    }


def test_radar_populate_registry_from_cached_api_response():
    """Verify RADAR repository publishes checksum under the archive filename."""
    repo = RadarRepository("10.60887/example", "https://datathek.oeaw.ac.at/radar/en/dataset/example")
    repo._api_response = {
        "10.60887-example.tar": {
            "url": "https://datathek.oeaw.ac.at/radar-backend/archives/example/versions/1/content",
            "checksum": "md5:ee7976db28d6ce0b88cfd3ef11560d7f",
            "size": 46069760,
        }
    }
    pup = SimpleNamespace(registry={})

    repo.populate_registry(pup)

    assert pup.registry == {"10.60887-example.tar": "md5:ee7976db28d6ce0b88cfd3ef11560d7f"}


def test_doi_resolution_warning_includes_doi_resolver_url(monkeypatch):
    """Verify DOI retry warning mentions the DOI resolver URL."""
    warnings = []
    debug_messages = []

    def fake_warning(message):
        warnings.append(message)

    def fake_debug(message):
        debug_messages.append(message)

    call_count = {"count": 0}

    def fake_doi_to_url(doi, timeout):
        call_count["count"] += 1
        if call_count["count"] == 1:
            raise Timeout("timed out")
        return "https://datathek.oeaw.ac.at/radar/en/dataset/example"

    monkeypatch.setattr(repositories_module, "doi_to_url", fake_doi_to_url)
    monkeypatch.setattr(repositories_module, "sleep", lambda seconds: None)
    monkeypatch.setattr(repositories_module.logger, "warning", fake_warning)
    monkeypatch.setattr(repositories_module.logger, "debug", fake_debug)
    monkeypatch.setattr(
        RadarRepository,
        "api_response",
        property(lambda self: {"10.60887-example.tar": {"url": "u", "checksum": "md5:x", "size": 1}}),
    )

    repo = doi_to_repository("10.60887/example")

    assert isinstance(repo, RadarRepository)
    assert warnings == [
        "Failed to resolve DOI 10.60887/example via https://doi.org/10.60887/example due to a connection or timeout error, retrying with exponential backoff..."
    ]
    assert any("Attempt 1/5 failed (Timeout)" in message for message in debug_messages)
