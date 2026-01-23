"""
Tests for the AudioArray extension.

This module contains comprehensive tests for the AudioDtype and AudioArray
classes that enable audio data processing in LOTUS.
"""

import base64
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lotus.dtype_extensions.audio import (
    AudioArray,
    AudioDtype,
    SUPPORTED_AUDIO_FORMATS,
)


class TestAudioDtype:
    """Tests for the AudioDtype class."""

    def test_dtype_name(self):
        """Should have correct name."""
        dtype = AudioDtype()
        assert dtype.name == "audio"

    def test_dtype_type(self):
        """Should have bytes as scalar type."""
        dtype = AudioDtype()
        assert dtype.type == bytes

    def test_na_value(self):
        """Should have None as na_value."""
        dtype = AudioDtype()
        assert dtype.na_value is None

    def test_construct_array_type(self):
        """Should return AudioArray type."""
        assert AudioDtype.construct_array_type() == AudioArray


class TestAudioArrayBasics:
    """Basic tests for AudioArray initialization and properties."""

    def test_initialization_with_paths(self):
        """Should initialize with file paths."""
        paths = ["audio1.wav", "audio2.mp3", "audio3.flac"]
        arr = AudioArray(paths)
        assert len(arr) == 3

    def test_initialization_with_none_values(self):
        """Should handle None values."""
        values = ["audio1.wav", None, "audio3.mp3"]
        arr = AudioArray(values)
        assert len(arr) == 3

    def test_dtype_property(self):
        """Should return AudioDtype instance."""
        arr = AudioArray(["test.wav"])
        assert isinstance(arr.dtype, AudioDtype)

    def test_empty_array(self):
        """Should handle empty arrays."""
        arr = AudioArray([])
        assert len(arr) == 0


class TestAudioArrayIndexing:
    """Tests for AudioArray indexing operations."""

    def test_single_item_access(self):
        """Should return single item for integer index."""
        paths = ["a.wav", "b.mp3", "c.flac"]
        arr = AudioArray(paths)
        assert arr[0] == "a.wav"
        assert arr[2] == "c.flac"

    def test_slice_access(self):
        """Should return AudioArray for slice."""
        paths = ["a.wav", "b.mp3", "c.flac", "d.ogg"]
        arr = AudioArray(paths)
        result = arr[1:3]
        assert isinstance(result, AudioArray)
        assert len(result) == 2

    def test_list_index_access(self):
        """Should return AudioArray for list of indices."""
        paths = ["a.wav", "b.mp3", "c.flac"]
        arr = AudioArray(paths)
        result = arr[[0, 2]]
        assert isinstance(result, AudioArray)
        assert len(result) == 2

    def test_setitem_single(self):
        """Should update single item."""
        arr = AudioArray(["a.wav", "b.mp3"])
        arr[0] = "new.wav"
        assert arr[0] == "new.wav"

    def test_setitem_slice(self):
        """Should update slice of items."""
        arr = AudioArray(["a.wav", "b.mp3", "c.flac"])
        arr[0:2] = ["x.wav", "y.mp3"]
        assert arr[0] == "x.wav"
        assert arr[1] == "y.mp3"


class TestAudioArrayMethods:
    """Tests for AudioArray methods."""

    def test_isna(self):
        """Should detect missing values."""
        arr = AudioArray(["a.wav", None, "c.flac"])
        result = arr.isna()
        assert result[0] is np.False_
        assert result[1] is np.True_
        assert result[2] is np.False_

    def test_take(self):
        """Should take elements by index."""
        arr = AudioArray(["a.wav", "b.mp3", "c.flac"])
        result = arr.take([2, 0])
        assert len(result) == 2
        assert result[0] == "c.flac"
        assert result[1] == "a.wav"

    def test_copy(self):
        """Should create a copy of the array."""
        arr = AudioArray(["a.wav", "b.mp3"])
        copy = arr.copy()
        assert len(copy) == len(arr)
        copy[0] = "modified.wav"
        assert arr[0] == "a.wav"  # Original unchanged

    def test_concat_same_type(self):
        """Should concatenate multiple AudioArrays."""
        arr1 = AudioArray(["a.wav", "b.mp3"])
        arr2 = AudioArray(["c.flac", "d.ogg"])
        result = AudioArray._concat_same_type([arr1, arr2])
        assert len(result) == 4

    def test_from_sequence(self):
        """Should construct from sequence."""
        paths = ["a.wav", "b.mp3"]
        result = AudioArray._from_sequence(paths)
        assert isinstance(result, AudioArray)
        assert len(result) == 2

    def test_to_numpy(self):
        """Should convert to numpy array."""
        arr = AudioArray(["a.wav", "b.mp3"])
        result = arr.to_numpy()
        assert isinstance(result, np.ndarray)
        assert len(result) == 2


class TestAudioArrayEquality:
    """Tests for AudioArray equality comparison."""

    def test_equality_with_audioarray(self):
        """Should compare with another AudioArray."""
        arr1 = AudioArray(["a.wav", "b.mp3"])
        arr2 = AudioArray(["a.wav", "c.flac"])
        result = arr1 == arr2
        assert result[0] is np.True_
        assert result[1] is np.False_

    def test_equality_with_list(self):
        """Should compare with list."""
        arr = AudioArray(["a.wav", "b.mp3"])
        result = arr == ["a.wav", "x.ogg"]
        assert result[0] is np.True_
        assert result[1] is np.False_

    def test_equality_with_scalar(self):
        """Should compare with scalar."""
        arr = AudioArray(["a.wav", "a.wav", "b.mp3"])
        result = arr == "a.wav"
        assert result[0] is np.True_
        assert result[1] is np.True_
        assert result[2] is np.False_


class TestAudioArrayRepr:
    """Tests for AudioArray string representation."""

    def test_repr_short_array(self):
        """Should show all elements for short arrays."""
        arr = AudioArray(["a.wav", "b.mp3"])
        repr_str = repr(arr)
        assert "AudioArray" in repr_str
        assert "a.wav" in repr_str

    def test_repr_with_none(self):
        """Should handle None values in repr."""
        arr = AudioArray(["a.wav", None])
        repr_str = repr(arr)
        assert "None" in repr_str

    def test_formatter(self):
        """Should return formatter function."""
        arr = AudioArray(["a.wav"])
        formatter = arr._formatter()
        assert callable(formatter)


class TestAudioArrayMimeType:
    """Tests for MIME type detection."""

    def test_wav_mime_type(self):
        """Should detect WAV MIME type."""
        arr = AudioArray(["test.wav"])
        assert arr.get_mime_type(0) == "audio/wav"

    def test_mp3_mime_type(self):
        """Should detect MP3 MIME type."""
        arr = AudioArray(["test.mp3"])
        assert arr.get_mime_type(0) == "audio/mpeg"

    def test_flac_mime_type(self):
        """Should detect FLAC MIME type."""
        arr = AudioArray(["test.flac"])
        assert arr.get_mime_type(0) == "audio/flac"

    def test_ogg_mime_type(self):
        """Should detect OGG MIME type."""
        arr = AudioArray(["test.ogg"])
        assert arr.get_mime_type(0) == "audio/ogg"

    def test_none_value_mime_type(self):
        """Should return None for None values."""
        arr = AudioArray([None])
        assert arr.get_mime_type(0) is None


class TestSupportedFormats:
    """Tests for supported audio formats."""

    def test_wav_supported(self):
        """WAV should be supported."""
        assert ".wav" in SUPPORTED_AUDIO_FORMATS

    def test_mp3_supported(self):
        """MP3 should be supported."""
        assert ".mp3" in SUPPORTED_AUDIO_FORMATS

    def test_flac_supported(self):
        """FLAC should be supported."""
        assert ".flac" in SUPPORTED_AUDIO_FORMATS

    def test_ogg_supported(self):
        """OGG should be supported."""
        assert ".ogg" in SUPPORTED_AUDIO_FORMATS

    def test_mp4_supported(self):
        """MP4 should be supported."""
        assert ".mp4" in SUPPORTED_AUDIO_FORMATS


class TestAudioArrayWithRealFiles:
    """Tests with real audio files (when available)."""

    def test_load_from_bytes(self):
        """Should handle raw bytes input."""
        audio_bytes = b"RIFF\x00\x00\x00\x00WAVEfmt "  # Minimal WAV header
        arr = AudioArray([audio_bytes])
        assert len(arr) == 1

    def test_base64_data_uri(self):
        """Should handle base64 data URIs."""
        audio_bytes = b"test audio data"
        encoded = base64.b64encode(audio_bytes).decode("utf-8")
        data_uri = f"data:audio/wav;base64,{encoded}"
        
        arr = AudioArray([data_uri])
        mime_type = arr.get_mime_type(0)
        assert mime_type == "audio/wav"


class TestPandasIntegration:
    """Tests for pandas DataFrame integration."""

    def test_series_with_audio_dtype(self):
        """Should work in pandas Series."""
        arr = AudioArray(["a.wav", "b.mp3", "c.flac"])
        series = pd.Series(arr)
        assert len(series) == 3

    def test_dataframe_column(self):
        """Should work as DataFrame column."""
        arr = AudioArray(["a.wav", "b.mp3"])
        df = pd.DataFrame({"audio": arr, "label": ["speech", "music"]})
        assert len(df) == 2
        assert "audio" in df.columns

    def test_nbytes(self):
        """Should calculate memory usage."""
        arr = AudioArray(["a.wav", "b.mp3"])
        assert arr.nbytes > 0
