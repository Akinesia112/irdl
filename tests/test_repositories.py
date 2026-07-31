"""Tests for repository adapters."""

import logging

import requests

from irdl import repositories
from irdl.repositories import DSPACE_API_ROOT, DSpaceRepository

HTTP_ERROR_STATUS = 400


class FakeResponse:
    """Small requests.Response stand-in for repository tests."""

    def __init__(self, payload: dict, status_code: int = 200, url: str = "https://example.test") -> None:
        self._payload = payload
        self.status_code = status_code
        self.url = url

    def json(self) -> dict:
        """Return the fake JSON payload."""
        return self._payload

    def raise_for_status(self) -> None:
        """Raise requests.HTTPError for non-2xx responses."""
        if self.status_code >= HTTP_ERROR_STATUS:
            msg = f"{self.status_code} for {self.url}"
            raise requests.HTTPError(msg, response=self)


class FakeSession:
    """Tiny fake session that serves pre-baked JSON responses."""

    def __init__(self, responses: dict[tuple[str, tuple[tuple[str, str], ...]], FakeResponse]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict | None]] = []

    def __enter__(self):
        """Return the fake session itself."""
        return self

    def __exit__(self, *_exc_info) -> None:
        """Match the requests.Session context-manager API."""

    def get(self, url: str, params: dict | None = None, timeout=None) -> FakeResponse:
        """Return the configured fake response for one GET request."""
        _ = timeout
        self.calls.append((url, params))
        key = (url, tuple(sorted((params or {}).items())))
        try:
            return self.responses[key]
        except KeyError as error:
            msg = f"Unexpected GET {url} params={params}"
            raise AssertionError(msg) from error


def _response_key(url: str, params: dict | None = None) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Build lookup keys for FakeSession."""
    return (url, tuple(sorted((params or {}).items())))


def test_dspace_repository_resolves_item_via_pid_find(monkeypatch):
    """Verify DSpaceRepository follows pid/find -> bundles -> bitstreams."""
    doi = "10.14279/depositonce-5718.5"
    bundles_url = f"{DSPACE_API_ROOT}/core/items/test-uuid/bundles"
    bitstreams_url = f"{DSPACE_API_ROOT}/core/bundles/original/bitstreams"
    session = FakeSession(
        {
            _response_key(DSPACE_API_ROOT): FakeResponse({"dspaceVersion": "DSpace 9.3"}),
            _response_key(f"{DSPACE_API_ROOT}/pid/find", {"id": doi}): FakeResponse(
                {"uuid": "test-uuid", "_links": {"bundles": {"href": bundles_url}}}
            ),
            _response_key(bundles_url): FakeResponse(
                {"_embedded": {"bundles": [{"name": "ORIGINAL", "_links": {"bitstreams": {"href": bitstreams_url}}}]}}
            ),
            _response_key(bitstreams_url): FakeResponse(
                {
                    "_embedded": {
                        "bitstreams": [
                            {
                                "name": "FABIAN_HRTF_DATABASE_v4.zip",
                                "sizeBytes": 123,
                                "checkSum": {"checkSumAlgorithm": "MD5", "value": "abc"},
                                "_links": {"content": {"href": f"{DSPACE_API_ROOT}/core/bitstreams/file/content"}},
                            }
                        ]
                    }
                }
            ),
        }
    )
    monkeypatch.setattr(repositories, "_make_session", lambda: session)

    repo = DSpaceRepository(doi=doi, archive_url="https://depositonce.tu-berlin.de/handle/11303/6153.5")

    assert repo.api_response == {
        "FABIAN_HRTF_DATABASE_v4.zip": {
            "url": f"{DSPACE_API_ROOT}/core/bitstreams/file/content",
            "checksum": "MD5:abc",
            "size": 123,
        }
    }
    assert (f"{DSPACE_API_ROOT}/pid/find", {"id": doi}) in session.calls
    assert not any(url.endswith("/6153.5/bundles") for url, _params in session.calls)


def test_dspace_repository_warns_on_version_mismatch(monkeypatch, caplog):
    """Verify unexpected DSpace versions emit a warning."""
    doi = "10.14279/depositonce-5718.5"
    bundles_url = f"{DSPACE_API_ROOT}/core/items/test-uuid/bundles"
    bitstreams_url = f"{DSPACE_API_ROOT}/core/bundles/original/bitstreams"
    session = FakeSession(
        {
            _response_key(DSPACE_API_ROOT): FakeResponse({"dspaceVersion": "DSpace 8.0"}),
            _response_key(f"{DSPACE_API_ROOT}/pid/find", {"id": doi}): FakeResponse(
                {"uuid": "test-uuid", "_links": {"bundles": {"href": bundles_url}}}
            ),
            _response_key(bundles_url): FakeResponse(
                {"_embedded": {"bundles": [{"name": "ORIGINAL", "_links": {"bitstreams": {"href": bitstreams_url}}}]}}
            ),
            _response_key(bitstreams_url): FakeResponse(
                {
                    "_embedded": {
                        "bitstreams": [
                            {
                                "name": "FABIAN_HRTF_DATABASE_v4.zip",
                                "sizeBytes": 123,
                                "checkSum": {"checkSumAlgorithm": "MD5", "value": "abc"},
                                "_links": {"content": {"href": f"{DSPACE_API_ROOT}/core/bitstreams/file/content"}},
                            }
                        ]
                    }
                }
            ),
        }
    )
    monkeypatch.setattr(repositories, "_make_session", lambda: session)

    repo = DSpaceRepository(doi=doi, archive_url="https://depositonce.tu-berlin.de/handle/11303/6153.5")
    with caplog.at_level(logging.WARNING, logger="irdl"):
        _ = repo.api_response

    assert "DepositOnce reports DSpace 8.0; expected DSpace 9.x" in caplog.text
