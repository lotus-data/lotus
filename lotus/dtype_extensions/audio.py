"""
Audio data type extension for LOTUS semantic operators.

This module provides a custom pandas ExtensionDtype and ExtensionArray for
storing and manipulating audio data within DataFrames. It enables semantic
operators to process audio files (.wav, .mp3, .mp4, .flac, .ogg) alongside
other data types.

The implementation mirrors the ImageArray pattern but is specialized for
audio content, supporting both file paths and base64-encoded audio data.

Example:
    >>> import pandas as pd
    >>> from lotus.dtype_extensions import AudioArray
    >>> 
    >>> audio_files = ['speech1.wav', 'speech2.mp3', 'music.flac']
    >>> df = pd.DataFrame({'audio': AudioArray(audio_files)})
    >>> df.sem_filter("the {audio} contains speech")
"""

import base64
import io
import os
import sys
from pathlib import Path
from typing import Any, Sequence, Union

import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionArray, ExtensionDtype


# Supported audio formats and their MIME types
SUPPORTED_AUDIO_FORMATS = {
    '.wav': 'audio/wav',
    '.mp3': 'audio/mpeg',
    '.mp4': 'audio/mp4',
    '.m4a': 'audio/mp4',
    '.flac': 'audio/flac',
    '.ogg': 'audio/ogg',
    '.webm': 'audio/webm',
}


class AudioDtype(ExtensionDtype):
    """
    A custom pandas ExtensionDtype for representing audio data.
    
    This dtype allows audio files or audio data to be stored in pandas
    DataFrames and processed by LOTUS semantic operators.
    
    Attributes:
        name: The string identifier for this dtype ("audio").
        type: The scalar type for this dtype (bytes for raw audio data).
        na_value: The missing value representation (None).
    """
    
    name = "audio"
    type = bytes
    na_value = None
    
    @classmethod
    def construct_array_type(cls):
        """
        Return the array type associated with this dtype.
        
        Returns:
            type: The AudioArray class.
        """
        return AudioArray


class AudioArray(ExtensionArray):
    """
    A pandas ExtensionArray for storing and manipulating audio data.
    
    This class allows audio files or audio data references to be stored
    in a pandas Series or DataFrame column. It supports efficient access,
    caching, and conversion to various formats for LLM processing.
    
    Attributes:
        _data: The underlying numpy array storing audio file paths or data.
        _dtype: The AudioDtype instance for this array.
        _cached_audio: Cache for loaded audio data, keyed by (index, format).
        
    Example:
        >>> audio_arr = AudioArray(['file1.wav', 'file2.mp3'])
        >>> len(audio_arr)
        2
        >>> audio_arr.get_audio(0, audio_format='base64')
        'data:audio/wav;base64,UklGRi...'
    """
    
    def __init__(self, values):
        """
        Initialize the AudioArray.
        
        Args:
            values: The initial values for the array. Can be file paths,
                    URLs, or base64-encoded audio strings.
        """
        self._data = np.asarray(values, dtype=object)
        self._dtype = AudioDtype()
        self._cached_audio: dict[tuple[int, str], Union[bytes, str, None]] = {}
    
    def __getitem__(self, item: Union[int, slice, Sequence[int]]) -> Any:
        """
        Retrieve one or more items from the array.
        
        Args:
            item: The index, slice, or sequence of indices to retrieve.
            
        Returns:
            The audio reference at the given index, or a new AudioArray
            for slices and sequences.
        """
        result = self._data[item]
        
        if isinstance(item, (int, np.integer)):
            return result
        
        return AudioArray(result)
    
    def __setitem__(
        self, 
        key: Union[int, slice, Sequence[int], np.ndarray], 
        value: Any
    ) -> None:
        """
        Set one or more values in the array, invalidating cache entries.
        
        Args:
            key: The index, slice, sequence, or boolean mask to set.
            value: The value or values to assign.
        """
        # Normalize key to a list of indices
        if isinstance(key, np.ndarray):
            if key.dtype == bool:
                key = np.where(key)[0]
            key = key.tolist()
        if isinstance(key, (int, np.integer)):
            key = [key]
        if isinstance(key, slice):
            key = range(*key.indices(len(self)))
        
        # Handle iterable values
        if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
            for idx, val in zip(key, value):
                self._data[idx] = val
                self._invalidate_cache(idx)
        else:
            for idx in key:
                self._data[idx] = value
                self._invalidate_cache(idx)
    
    def _invalidate_cache(self, idx: int) -> None:
        """
        Remove cached audio data for the specified index.
        
        Args:
            idx: The index to invalidate in the cache.
        """
        keys_to_remove = [k for k in self._cached_audio if k[0] == idx]
        for key in keys_to_remove:
            del self._cached_audio[key]
    
    def get_audio(
        self, 
        idx: int, 
        audio_format: str = "base64"
    ) -> Union[bytes, str, None]:
        """
        Fetch and return audio data for the given index.
        
        Supports caching to avoid repeated file reads or conversions.
        
        Args:
            idx: The index of the audio to fetch.
            audio_format: The format to return ("base64" or "bytes").
            
        Returns:
            The audio data in the requested format, or None if unavailable.
            
        Raises:
            ValueError: If the audio format is not supported.
        """
        cache_key = (idx, audio_format)
        
        if cache_key in self._cached_audio:
            return self._cached_audio[cache_key]
        
        raw_value = self._data[idx]
        if raw_value is None:
            return None
        
        # Load and convert the audio
        audio_data = self._load_audio(raw_value, audio_format)
        self._cached_audio[cache_key] = audio_data
        
        return audio_data
    
    def _load_audio(
        self, 
        value: Any, 
        audio_format: str
    ) -> Union[bytes, str, None]:
        """
        Load audio data from various source types.
        
        Args:
            value: The audio source (file path, URL, or base64 string).
            audio_format: The desired output format.
            
        Returns:
            The audio data in the requested format.
        """
        if value is None:
            return None
        
        # Handle file paths
        if isinstance(value, (str, Path)):
            path = Path(value)
            if path.exists() and path.is_file():
                return self._load_from_file(path, audio_format)
            
            # Check if it's already a base64 data URI
            if isinstance(value, str) and value.startswith("data:audio"):
                if audio_format == "base64":
                    return value
                return self._decode_base64_uri(value)
        
        # Handle raw bytes
        if isinstance(value, bytes):
            if audio_format == "bytes":
                return value
            return self._encode_to_base64(value, "audio/octet-stream")
        
        return None
    
    def _load_from_file(
        self, 
        path: Path, 
        audio_format: str
    ) -> Union[bytes, str, None]:
        """
        Load audio data from a file path.
        
        Args:
            path: The path to the audio file.
            audio_format: The desired output format.
            
        Returns:
            The audio data in the requested format.
        """
        suffix = path.suffix.lower()
        mime_type = SUPPORTED_AUDIO_FORMATS.get(suffix, "audio/octet-stream")
        
        try:
            with open(path, "rb") as f:
                audio_bytes = f.read()
            
            if audio_format == "bytes":
                return audio_bytes
            
            return self._encode_to_base64(audio_bytes, mime_type)
            
        except (IOError, OSError) as e:
            # Log the error but don't crash - return None for missing files
            return None
    
    def _encode_to_base64(self, audio_bytes: bytes, mime_type: str) -> str:
        """
        Encode raw audio bytes to a base64 data URI.
        
        Args:
            audio_bytes: The raw audio data.
            mime_type: The MIME type of the audio.
            
        Returns:
            A base64-encoded data URI string.
        """
        encoded = base64.b64encode(audio_bytes).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"
    
    def _decode_base64_uri(self, uri: str) -> bytes:
        """
        Decode a base64 data URI to raw bytes.
        
        Args:
            uri: The base64 data URI string.
            
        Returns:
            The decoded raw audio bytes.
        """
        # Extract the base64 portion after the header
        if ";base64," in uri:
            _, encoded = uri.split(";base64,", 1)
            return base64.b64decode(encoded)
        return b""
    
    def isna(self) -> np.ndarray:
        """
        Detect missing values in the array.
        
        Returns:
            Boolean array indicating missing values.
        """
        return pd.isna(self._data)
    
    def take(
        self, 
        indices: Sequence[int], 
        allow_fill: bool = False, 
        fill_value=None
    ) -> "AudioArray":
        """
        Take elements from the array by index.
        
        Args:
            indices: Indices of elements to take.
            allow_fill: If True, -1 indices indicate fill positions.
            fill_value: Value to use for fill positions.
            
        Returns:
            A new AudioArray with the selected elements.
        """
        result = self._data.take(indices, axis=0)
        if allow_fill and fill_value is not None:
            result[np.asarray(indices) == -1] = fill_value
        return AudioArray(result)
    
    def copy(self) -> "AudioArray":
        """
        Return a copy of the array, including the cache.
        
        Returns:
            A new AudioArray with copied data.
        """
        new_array = AudioArray(self._data.copy())
        new_array._cached_audio = self._cached_audio.copy()
        return new_array
    
    @classmethod
    def _concat_same_type(cls, to_concat: Sequence["AudioArray"]) -> "AudioArray":
        """
        Concatenate multiple AudioArray instances.
        
        Args:
            to_concat: Sequence of AudioArray instances to concatenate.
            
        Returns:
            A new AudioArray containing all elements.
        """
        combined_data = np.concatenate([arr._data for arr in to_concat])
        return cls._from_sequence(combined_data)
    
    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        """
        Construct a new AudioArray from a sequence of scalars.
        
        Args:
            scalars: The input sequence of audio references.
            dtype: Ignored (for API compatibility).
            copy: If True, copy the input data.
            
        Returns:
            A new AudioArray instance.
        """
        if copy:
            scalars = np.array(scalars, dtype=object, copy=True)
        return cls(scalars)
    
    def __len__(self) -> int:
        """Return the number of elements in the array."""
        return len(self._data)
    
    def __eq__(self, other) -> np.ndarray:  # type: ignore
        """
        Compare this AudioArray to another object for equality.
        
        Args:
            other: Another AudioArray, sequence, or scalar to compare.
            
        Returns:
            Boolean array indicating element-wise equality.
        """
        if isinstance(other, AudioArray):
            return self._data == other._data
        
        if hasattr(other, "__iter__") and not isinstance(other, str):
            if len(other) != len(self):
                return np.repeat(False, len(self))
            return np.array([a == b for a, b in zip(self._data, other)], dtype=bool)
        
        return np.array([a == other for a in self._data], dtype=bool)
    
    @property
    def dtype(self) -> AudioDtype:
        """Return the dtype for this array."""
        return self._dtype
    
    @property
    def nbytes(self) -> int:
        """
        Return the total memory consumption of the array elements.
        
        Returns:
            Total bytes consumed by the stored audio references.
        """
        return sum(sys.getsizeof(item) for item in self._data if item)
    
    def __repr__(self) -> str:
        """Return a string representation of the AudioArray."""
        preview = ", ".join([
            f"<Audio: {self._get_audio_info(item)}>" if item else "None"
            for item in self._data[:5]
        ])
        suffix = ", ..." if len(self._data) > 5 else ""
        return f"AudioArray([{preview}{suffix}])"
    
    def _get_audio_info(self, item: Any) -> str:
        """
        Get a brief description of an audio item.
        
        Args:
            item: The audio reference to describe.
            
        Returns:
            A short string describing the audio item.
        """
        if isinstance(item, (str, Path)):
            path = Path(item)
            if path.suffix.lower() in SUPPORTED_AUDIO_FORMATS:
                return path.name
            if str(item).startswith("data:audio"):
                return "base64"
        if isinstance(item, bytes):
            return f"{len(item)} bytes"
        return str(type(item).__name__)
    
    def _formatter(self, boxed: bool = False):
        """
        Return a formatter function for displaying array elements.
        
        Args:
            boxed: Whether to use a boxed formatter (unused).
            
        Returns:
            A function that formats an element for display.
        """
        return lambda x: f"<Audio: {self._get_audio_info(x)}>" if x else "None"
    
    def to_numpy(self, dtype=None, copy=False, na_value=None) -> np.ndarray:
        """
        Convert the AudioArray to a numpy array.
        
        Args:
            dtype: Ignored (for API compatibility).
            copy: If True, return a copy of the data.
            na_value: Ignored (for API compatibility).
            
        Returns:
            A numpy array of audio references.
        """
        if copy:
            return self._data.copy()
        return self._data
    
    def __array__(self, dtype=None) -> np.ndarray:
        """
        Numpy array interface for AudioArray.
        
        Args:
            dtype: Ignored (for API compatibility).
            
        Returns:
            A numpy array of audio references.
        """
        return self.to_numpy(dtype=dtype)
    
    def get_duration(self, idx: int) -> float | None:
        """
        Get the duration of an audio file in seconds.
        
        This method attempts to read the duration without loading
        the entire audio file into memory when possible.
        
        Args:
            idx: The index of the audio to get duration for.
            
        Returns:
            Duration in seconds, or None if unavailable.
            
        Note:
            This requires the audio file to be accessible on disk.
            Duration detection may not work for all formats.
        """
        raw_value = self._data[idx]
        if raw_value is None:
            return None
        
        if isinstance(raw_value, (str, Path)):
            path = Path(raw_value)
            if path.exists():
                # Basic duration estimation based on file size
                # More accurate duration requires format-specific libraries
                return None  # Placeholder for format-specific implementation
        
        return None
    
    def get_mime_type(self, idx: int) -> str | None:
        """
        Get the MIME type of an audio file.
        
        Args:
            idx: The index of the audio to get MIME type for.
            
        Returns:
            The MIME type string, or None if unavailable.
        """
        raw_value = self._data[idx]
        if raw_value is None:
            return None
        
        if isinstance(raw_value, (str, Path)):
            path = Path(raw_value)
            suffix = path.suffix.lower()
            return SUPPORTED_AUDIO_FORMATS.get(suffix)
        
        if isinstance(raw_value, str) and raw_value.startswith("data:"):
            # Extract MIME type from data URI
            if ";" in raw_value:
                return raw_value.split(";")[0].replace("data:", "")
        
        return None
