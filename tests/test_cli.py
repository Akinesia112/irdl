"""Tests for CLI wrapper behavior."""

from inspect import signature
from pathlib import Path

import typer

from irdl.cli import _make_wrapper

RESULT_DIR = Path("tests/_tmp_cli")


class DummyDataset:
    """Concrete stand-in for CLI wrapper tests."""

    name = "dummy"

    @classmethod
    def get(cls, output_format: str = "pyfar"):
        """Return representative values for different output formats."""
        if output_format in {"raw", "sofa", "hdf5"}:
            suffix = "h5" if output_format == "hdf5" else "sofa"
            return RESULT_DIR / f"result.{suffix}"
        return {"output_format": output_format}


def test_make_wrapper_echoes_returned_path(monkeypatch):
    """Verify file-based CLI results are printed to stdout."""
    echoed: list[object] = []
    monkeypatch.setattr(typer, "echo", echoed.append)

    wrapper = _make_wrapper(
        DummyDataset,
        DummyDataset.get,
        signature(DummyDataset.get.__func__).parameters,
        "help",
        DummyDataset.name,
        {},
    )

    result = wrapper(output_format="sofa")

    assert result == RESULT_DIR / "result.sofa"
    # typer.style adds ANSI color codes (\x1b[96m = bright cyan, \x1b[0m = reset)
    assert echoed == [f"\x1b[96m{RESULT_DIR / 'result.sofa'}\x1b[0m"]


def test_make_wrapper_echoes_formatted_dict(monkeypatch):
    """Verify in-memory CLI results are rendered as formatted text."""
    echoed: list[object] = []
    monkeypatch.setattr(typer, "echo", echoed.append)

    wrapper = _make_wrapper(
        DummyDataset,
        DummyDataset.get,
        signature(DummyDataset.get.__func__).parameters,
        "help",
        DummyDataset.name,
        {},
    )

    result = wrapper(output_format="numpy")

    assert result == {"output_format": "numpy"}
    # typer.style adds ANSI color codes (\x1b[96m = bright cyan, \x1b[0m = reset)
    assert echoed == [f"\x1b[96moutput_format:\n  numpy\x1b[0m"]
