"""Tests for Dataset conversion methods."""

import tempfile
from pathlib import Path

import numpy as np
import sofar as sf

from irdl.base import BaseDataset


# Create a concrete Dataset subclass for testing conversion methods
# (BaseDataset is abstract and cannot be instantiated directly)
class TestDataset(BaseDataset):
    """Concrete Dataset subclass for testing conversion methods."""

    name = "test"
    doi = "10.0000/test"

    def validate_params(self, dataset_kwargs):
        """No-op validation for test dataset."""
        pass

    def download(self, **kwargs):
        """No-op download for test dataset."""
        pass

    def ingest(self, file_path):
        """No-op ingest for test dataset."""
        pass

    def _source_filename(self, **kwargs):
        """Return test file name."""
        return "test.sofa"


# Create the TestDataset instance to use for all conversion tests
test_dataset = TestDataset()


class TestConversionToPyFar:
    """Tests for _to_pyfar conversion method."""

    def test_conversion_to_pyfar_returns_dict(self, sofa_object):
        """Verify _to_pyfar returns a dictionary."""
        result = test_dataset._to_pyfar(sofa_object)
        assert isinstance(result, dict)

    def test_conversion_to_pyfar_structure(self, sofa_object):
        """Verify _to_pyfar returns dict with expected keys."""
        result = test_dataset._to_pyfar(sofa_object)
        expected_keys = {"impulse_response", "source_coordinates", "receiver_coordinates"}
        assert set(result.keys()) == expected_keys

    def test_conversion_to_pyfar_types(self, sofa_object):
        """Verify _to_pyfar returns pyfar objects of correct types."""
        import pyfar as pf

        result = test_dataset._to_pyfar(sofa_object)

        assert isinstance(result["impulse_response"], pf.Signal)
        assert isinstance(result["source_coordinates"], pf.Coordinates)
        assert isinstance(result["receiver_coordinates"], pf.Coordinates)

    def test_conversion_to_pyfar_sampling_rate(self, sofa_object):
        """Verify _to_pyfar preserves sampling rate."""
        result = test_dataset._to_pyfar(sofa_object)
        assert result["impulse_response"].sampling_rate == sofa_object.Data_SamplingRate


class TestConversionToNumpy:
    """Tests for _to_numpy conversion method."""

    def test_conversion_to_numpy_returns_dict(self, sofa_object):
        """Verify _to_numpy returns a dictionary."""
        result = test_dataset._to_numpy(sofa_object)
        assert isinstance(result, dict)

    def test_conversion_to_numpy_structure(self, sofa_object):
        """Verify _to_numpy returns dict with expected keys."""
        result = test_dataset._to_numpy(sofa_object)
        expected_keys = {
            "impulse_response",
            "source_coordinates",
            "receiver_coordinates",
            "sampling_rate",
        }
        assert set(result.keys()) == expected_keys

    def test_conversion_to_numpy_types(self, sofa_object):
        """Verify _to_numpy returns numpy arrays."""
        result = test_dataset._to_numpy(sofa_object)

        assert isinstance(result["impulse_response"], np.ndarray)
        assert isinstance(result["source_coordinates"], np.ndarray)
        assert isinstance(result["receiver_coordinates"], np.ndarray)
        assert isinstance(result["sampling_rate"], float)

    def test_conversion_to_numpy_values(self, sofa_object):
        """Verify _to_numpy preserves data values."""
        result = test_dataset._to_numpy(sofa_object)

        # Check IR data matches
        np.testing.assert_array_equal(
            result["impulse_response"],
            sofa_object.Data_IR,
        )
        # Check sampling rate
        assert result["sampling_rate"] == sofa_object.Data_SamplingRate


class TestConversionToSofa:
    """Tests for _to_sofa conversion method."""

    def test_conversion_to_sofa_returns_path(self, sofa_object):
        """Verify _to_sofa returns a Path object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = test_dataset._to_sofa(sofa_object, Path(tmpdir) / "test.sofa")
            assert isinstance(result, Path)
            assert result.exists()

    def test_conversion_to_sofa_writable(self, sofa_object, tmp_path):
        """Verify _to_sofa writes a valid SOFA file."""
        result = test_dataset._to_sofa(sofa_object, Path(tmp_path) / "test.sofa")

        # Verify file can be read back
        loaded_sofa = sf.read_sofa(str(result))
        assert loaded_sofa.Data_IR.shape == sofa_object.Data_IR.shape
        assert loaded_sofa.Data_SamplingRate == sofa_object.Data_SamplingRate


class TestConversionToHdf5:
    """Tests for _to_hdf5 conversion method."""

    def test_conversion_to_hdf5_returns_path(self, sofa_object):
        """Verify _to_hdf5 returns a Path object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = test_dataset._to_hdf5(sofa_object, Path(tmpdir) / "test.h5")
            assert isinstance(result, Path)
            assert result.exists()

    def test_conversion_to_hdf5_structure(self, sofa_object):
        """Verify _to_hdf5 writes file with expected HDF5 structure."""
        import h5py

        with tempfile.TemporaryDirectory() as tmpdir:
            result = test_dataset._to_hdf5(sofa_object, Path(tmpdir) / "test.h5")

            with h5py.File(result, "r") as f:
                # Check data group exists
                assert "data" in f
                assert "impulse_response" in f["data"]
                assert "location" in f["data"]
                assert "source" in f["data/location"]
                assert "receiver" in f["data/location"]

                # Check metadata group exists
                assert "metadata" in f
                assert "sampling_rate" in f["metadata"]

    def test_conversion_to_hdf5_optional_fields(self, sofa_object):
        """Verify _to_hdf5 includes optional temperature/humidity fields when present."""
        import h5py

        # Manually add optional fields to the sofa object
        # Note: This bypasses SOFA's protected mode for testing purposes
        sofa_object._protected = False
        sofa_object.Data_Temperature = np.array([20.0, 20.0])
        sofa_object.Data_Humidity = np.array([50.0, 50.0])

        with tempfile.TemporaryDirectory() as tmpdir:
            result = test_dataset._to_hdf5(sofa_object, Path(tmpdir) / "test.h5")

            with h5py.File(result, "r") as f:
                assert "temperature" in f["metadata"]
                assert "humidity" in f["metadata"]


class TestConversionRoundtrip:
    """Tests for roundtrip conversion (SOFA -> conversion -> SOFA)."""

    def test_pyfar_roundtrip(self, sofa_object):
        """Verify SOFA -> pyfar -> SOFA preserves data."""
        # Convert to pyfar
        pyfar_result = test_dataset._to_pyfar(sofa_object)

        # Convert back to SOFA via pyfar
        # Note: pyfar.Signal can be converted back to numpy, but not directly to SOFA
        # This test verifies the data is preserved in the pyfar representation
        assert pyfar_result["impulse_response"].time.shape == sofa_object.Data_IR.shape
        assert pyfar_result["impulse_response"].sampling_rate == sofa_object.Data_SamplingRate


class TestConversionEdgeCases:
    """Tests for edge cases in conversion methods that would have caught the bug."""

    def test_pyfar_with_3d_receiver_position(self):
        """Test _to_pyfar with 3D ReceiverPosition array (like FABIAN dataset).
        
        This would have caught the original bug where 3D coordinate arrays
        with shape (N, 3, 1) caused unpacking errors.
        """
        import pyfar as pf
        
        # Create SOFA object with 3D ReceiverPosition (shape: N, 3, 1)
        sofa = sf.Sofa("GeneralFIR")
        sofa.Data_IR = np.random.randn(2, 2, 256).astype(np.float32)
        sofa.Data_SamplingRate = 44100
        sofa.SourcePosition = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)  # Shape: (2, 3)
        sofa.ReceiverPosition = np.array([[[0.0], [0.5], [0.0]], [[0.0], [-0.5], [0.0]]], dtype=np.float64)  # Shape: (2, 3, 1)
        sofa.Data_Delay = np.zeros(2, dtype=np.float32)
        
        # This should not raise ValueError about shape mismatch
        result = test_dataset._to_pyfar(sofa)
        
        assert isinstance(result["receiver_coordinates"], pf.Coordinates)
        assert result["receiver_coordinates"].cshape == (2,)  # Should have 2 receivers

    def test_pyfar_with_3d_source_position(self):
        """Test _to_pyfar with 3D SourcePosition array.
        
        Ensures both source and receiver 3D arrays are handled.
        """
        import pyfar as pf
        
        # Create SOFA object with 3D SourcePosition (shape: N, 3, 1)
        sofa = sf.Sofa("GeneralFIR")
        sofa.Data_IR = np.random.randn(2, 2, 256).astype(np.float32)
        sofa.Data_SamplingRate = 44100
        sofa.SourcePosition = np.array([[[0.0], [0.0], [0.0]], [[1.0], [0.0], [0.0]]], dtype=np.float64)  # Shape: (2, 3, 1)
        sofa.ReceiverPosition = np.array([[0.0, 0.5, 0.0], [0.0, -0.5, 0.0]], dtype=np.float64)  # Shape: (2, 3)
        sofa.Data_Delay = np.zeros(2, dtype=np.float32)
        
        result = test_dataset._to_pyfar(sofa)
        
        assert isinstance(result["source_coordinates"], pf.Coordinates)
        assert result["source_coordinates"].cshape == (2,)  # Should have 2 sources

    def test_pyfar_with_2d_sampling_rate_array(self):
        """Test _to_pyfar with 2D Data_SamplingRate array (like MIRACLE/SRIRACHA datasets).
        
        This would have caught the bug where multi-dimensional sampling rate
        arrays caused broadcasting errors in pyfar.Signal.
        """
        # Create SOFA object with 2D sampling rate array
        sofa = sf.Sofa("GeneralFIR")
        sofa.Data_IR = np.random.randn(2, 2, 256).astype(np.float32)
        sofa.Data_SamplingRate = np.array([[44100, 44100]])  # Shape: (1, 2)
        sofa.SourcePosition = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        sofa.ReceiverPosition = np.array([[0.0, 0.5, 0.0], [0.0, -0.5, 0.0]], dtype=np.float64)
        sofa.Data_Delay = np.zeros(2, dtype=np.float32)
        
        # This should not raise ValueError about broadcasting
        result = test_dataset._to_pyfar(sofa)
        
        # Should extract scalar sampling rate from the array
        assert result["impulse_response"].sampling_rate == 44100

    def test_pyfar_with_varying_sampling_rates_warns(self):
        """Test that _to_pyfar warns when multiple different sampling rates are found.
        
        Ensures user is notified when data may be inconsistent.
        """
        import warnings
        
        # Create SOFA object with varying sampling rates
        sofa = sf.Sofa("GeneralFIR")
        sofa.Data_IR = np.random.randn(2, 2, 256).astype(np.float32)
        sofa.Data_SamplingRate = np.array([[44100, 48000]])  # Shape: (1, 2) with different rates
        sofa.SourcePosition = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        sofa.ReceiverPosition = np.array([[0.0, 0.5, 0.0], [0.0, -0.5, 0.0]], dtype=np.float64)
        sofa.Data_Delay = np.zeros(2, dtype=np.float32)
        
        # Should raise UserWarning about multiple sampling rates
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = test_dataset._to_pyfar(sofa)
            
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "Multiple sampling rates found" in str(w[0].message)
            assert "44100" in str(w[0].message) or "48000" in str(w[0].message)
            assert "Using 44100 Hz" in str(w[0].message)  # First unique value is used
        
        # Should still use the first unique value
        assert result["impulse_response"].sampling_rate == 44100
