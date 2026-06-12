"""Tests for CLI result formatting helpers."""

import numpy as np

from irdl.cli import _format_cli_item, _format_cli_value


def test_format_cli_value_formats_numpy_mapping_readably():
    """Verify numpy outputs are rendered with keys and array summaries."""
    formatted = _format_cli_value(
        {
            "impulse_response": np.arange(12).reshape(3, 4),
            "sampling_rate": 44100.0,
        }
    )

    assert "impulse_response:" in formatted
    assert "ndarray shape=(3, 4) dtype=int64" in formatted
    assert "sampling_rate:" in formatted
    assert "44100.0" in formatted


def test_format_cli_item_truncates_large_numpy_arrays():
    """Verify large numpy arrays are summarized instead of dumped in full."""
    formatted = _format_cli_item(np.arange(100))

    assert "ndarray shape=(100,) dtype=int64" in formatted
    assert "..." in formatted
