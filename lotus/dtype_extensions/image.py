import sys
from typing import Any, Sequence, Union

import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionArray, ExtensionDtype
from PIL import Image

from lotus.utils import fetch_image


class ImageDtype(ExtensionDtype):
    """
    A custom pandas ExtensionDtype for representing images.

    Attributes:
        name (str): The string name for this dtype ("image").
        type (type): The scalar type for this dtype (PIL.Image.Image).
        na_value: The default missing value for this dtype (None).
    """

    name = "image"
    type = Image.Image
    na_value = None

    @classmethod
    def construct_array_type(cls):
        """
        Return the array type associated with this dtype.

        Returns:
            type: The ImageArray class.
        """
        return ImageArray


class ImageArray(ExtensionArray):
    """
    A pandas ExtensionArray for storing and manipulating images.

    This class allows images (or image references) to be stored in a pandas Series or DataFrame column,
    supporting efficient access, caching, and conversion to numpy arrays.

    Attributes:
        _data (np.ndarray): The underlying data array storing image objects or references.
        _dtype (ImageDtype): The dtype instance for this array.
        allowed_image_types (list): List of allowed image types for fetching.
        _cached_images (dict): Cache for loaded images, keyed by (index, image_type).
    """

    def __init__(self, values):
        """
        Initialize the ImageArray.

        Args:
            values (array-like): The initial values for the array. Can be images, file paths, or base64 strings.
        """
        self._data = np.asarray(values, dtype=object)
        self._dtype = ImageDtype()
        self.allowed_image_types = ["Image", "base64"]
        self._cached_images: dict[tuple[int, str], str | Image.Image | None] = {}  # Cache for loaded images

    def __getitem__(self, item: int | slice | Sequence[int]) -> np.ndarray:
        """
        Retrieve one or more items from the array.

        Args:
            item (int, slice, or sequence of int): The index or indices to retrieve.

        Returns:
            object: The image or reference at the given index, or a new ImageArray for slices/sequences.
        """
        result = self._data[item]

        if isinstance(item, (int, np.integer)):
            # Return the raw value for display purposes
            return result

        return ImageArray(result)

    def __setitem__(self, key: int | slice | Sequence[int] | np.ndarray, value: Any) -> None:
        """
        Set one or more values in the array, with cache invalidation.

        Args:
            key (int, slice, sequence, or boolean mask): The index/indices to set.
            value: The value(s) to assign.
        """
        if isinstance(key, np.ndarray):
            if key.dtype == bool:
                key = np.where(key)[0]
            key = key.tolist()
        if isinstance(key, (int, np.integer)):
            key = [key]
        if isinstance(key, slice):
            key = range(*key.indices(len(self)))
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
        Remove an item from the image cache.

        Args:
            idx (int): The index of the item to invalidate in the cache.
        """
        for image_type in self.allowed_image_types:
            if (idx, image_type) in self._cached_images:
                del self._cached_images[(idx, image_type)]

    def get_image(self, idx: int, image_type: str = "Image") -> Union[Image.Image, str, None]:
        """
        Fetch and return the actual image for a given index and type, using cache if available.

        Args:
            idx (int): The index of the image to fetch.
            image_type (str): The type of image to fetch ("Image" or "base64").

        Returns:
            Image.Image, str, or None: The loaded image, base64 string, or None if not available.
        """
        if (idx, image_type) not in self._cached_images:
            image_result = fetch_image(self._data[idx], image_type)
            assert image_result is None or isinstance(image_result, (Image.Image, str))
            self._cached_images[(idx, image_type)] = image_result
        return self._cached_images[(idx, image_type)]

    def isna(self) -> np.ndarray:
        """
        Detect missing values in the array.

        Returns:
            np.ndarray: Boolean array indicating missing values.
        """
        return pd.isna(self._data)

    def take(self, indices: Sequence[int], allow_fill: bool = False, fill_value=None) -> "ImageArray":
        """
        Take elements from the array by index.

        Args:
            indices (sequence of int): Indices to take.
            allow_fill (bool): If True, -1 in indices indicates missing values.
            fill_value: Value to use for missing values if allow_fill is True.

        Returns:
            ImageArray: A new ImageArray with the selected elements.
        """
        result = self._data.take(indices, axis=0)
        if allow_fill and fill_value is not None:
            result[indices == -1] = fill_value
        return ImageArray(result)

    def copy(self) -> "ImageArray":
        """
        Return a (shallow) copy of the array, including the cache.

        Returns:
            ImageArray: A copy of the current ImageArray.
        """
        new_array = ImageArray(self._data.copy())
        new_array._cached_images = self._cached_images.copy()
        return new_array

    def _concat_same_type(cls, to_concat: Sequence["ImageArray"]) -> "ImageArray":
        """
        Concatenate multiple ImageArray instances into a single one.

        Args:
            to_concat (Sequence[ImageArray]): A sequence of ImageArray instances to concatenate.

        Returns:
            ImageArray: A new ImageArray containing all elements from the input arrays.
        """
        # create list of all data
        combined_data = np.concatenate([arr._data for arr in to_concat])
        return cls._from_sequence(combined_data)

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        """
        Construct a new ImageArray from a sequence of scalars.

        Args:
            scalars (sequence): The input sequence of image objects or references.
            dtype: Ignored (for compatibility).
            copy (bool): If True, copy the input data.

        Returns:
            ImageArray: The constructed ImageArray.
        """
        if copy:
            scalars = np.array(scalars, dtype=object, copy=True)
        return cls(scalars)

    def __len__(self) -> int:
        """
        Return the number of elements in the array.

        Returns:
            int: The length of the array.
        """
        return len(self._data)

    def __eq__(self, other) -> np.ndarray:  # type: ignore
        """
        Compare this ImageArray to another object for equality.

        Args:
            other: Another ImageArray, sequence, or scalar to compare.

        Returns:
            np.ndarray: Boolean array indicating elementwise equality.
        """
        if isinstance(other, ImageArray):
            return np.array([_compare_images(img1, img2) for img1, img2 in zip(self._data, other._data)], dtype=bool)

        if hasattr(other, "__iter__") and not isinstance(other, str):
            if len(other) != len(self):
                return np.repeat(False, len(self))
            return np.array([_compare_images(img1, img2) for img1, img2 in zip(self._data, other)], dtype=bool)
        return np.array([_compare_images(img, other) for img in self._data], dtype=bool)

    @property
    def dtype(self) -> ImageDtype:
        """
        Return the dtype for this array.

        Returns:
            ImageDtype: The dtype instance.
        """
        return self._dtype

    @property
    def nbytes(self) -> int:
        """
        Return the total number of bytes consumed by the elements of the array.

        Returns:
            int: The total number of bytes.
        """
        return sum(sys.getsizeof(img) for img in self._data if img)

    def __repr__(self) -> str:
        """
        Return a string representation of the ImageArray.

        Returns:
            str: The string representation.
        """
        return f"ImageArray([{', '.join([f'<Image: {type(img)}>' if img is not None else 'None' for img in self._data[:5]])}, ...])"

    def _formatter(self, boxed: bool = False):
        """
        Return a formatter function for displaying array elements.

        Args:
            boxed (bool): Whether to use a boxed formatter (unused).

        Returns:
            callable: A function that formats an element for display.
        """
        return lambda x: f"<Image: {type(x)}>" if x is not None else "None"

    def to_numpy(self, dtype=None, copy=False, na_value=None) -> np.ndarray:
        """
        Convert the ImageArray to a numpy array of PIL Images.

        Args:
            dtype: Ignored (for compatibility).
            copy (bool): If True, return a copy of the data.
            na_value: Ignored (for compatibility).

        Returns:
            np.ndarray: A numpy array of PIL Images or None.
        """
        pil_images = []
        for i, img_data in enumerate(self._data):
            if isinstance(img_data, np.ndarray):
                image = self.get_image(i)
                pil_images.append(image)
            else:
                pil_images.append(img_data)
        result = np.empty(len(self), dtype=object)
        result[:] = pil_images
        return result

    def __array__(self, dtype=None) -> np.ndarray:
        """
        Numpy array interface for ImageArray.

        Args:
            dtype: Ignored (for compatibility).

        Returns:
            np.ndarray: A numpy array of PIL Images or None.
        """
        return self.to_numpy(dtype=dtype)


def _compare_images(img1, img2) -> bool:
    """
    Compare two images or image references for equality.

    Args:
        img1: The first image or reference.
        img2: The second image or reference.

    Returns:
        bool: True if the images are considered equal, False otherwise.
    """
    if img1 is None or img2 is None:
        return img1 is img2

    # Only fetch images when actually comparing
    if isinstance(img1, Image.Image) or isinstance(img2, Image.Image):
        img1 = fetch_image(img1)
        img2 = fetch_image(img2)
        return img1.size == img2.size and img1.mode == img2.mode and img1.tobytes() == img2.tobytes()
    else:
        return img1 == img2
