import sys
from typing import Sequence, Union

import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionArray, ExtensionDtype
from PIL import Image

from lotus.utils import fetch_image


class ImageDtype(ExtensionDtype):
    name = "image"
    type = Image.Image
    na_value = None

    @classmethod
    def construct_array_type(cls):
        return ImageArray


class ImageArray(ExtensionArray):
    def __init__(self, values):
        self._data = np.asarray(values, dtype=object)
        self._dtype = ImageDtype()
        self.allowed_image_types = ["Image", "base64"]
        self._cached_images: dict[tuple[int, str], str | Image.Image | None] = {}  # Cache for loaded images

    def __getitem__(self, item: int | slice | Sequence[int]) -> np.ndarray:
        result = self._data[item]

        if isinstance(item, (int, np.integer)):
            # Return the raw value for display purposes
            return result

        return ImageArray(result)

    def __setitem__(self, key, value) -> None:
        """Set one or more values inplace, with cache invalidation."""
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
        """Remove an item from the cache."""
        for image_type in self.allowed_image_types:
            if (idx, image_type) in self._cached_images:
                del self._cached_images[(idx, image_type)]

    def get_image(self, idx: int, image_type: str = "Image") -> Union[Image.Image, str, None]:
        """Explicit method to fetch and return the actual image"""
        if (idx, image_type) not in self._cached_images:
            image_result = fetch_image(self._data[idx], image_type)
            assert image_result is None or isinstance(image_result, (Image.Image, str))
            self._cached_images[(idx, image_type)] = image_result
        return self._cached_images[(idx, image_type)]

    def isna(self) -> np.ndarray:
        return pd.isna(self._data)

    def take(self, indices: Sequence[int], allow_fill: bool = False, fill_value=None) -> "ImageArray":
        result = self._data.take(indices, axis=0)
        if allow_fill and fill_value is not None:
            result[indices == -1] = fill_value
        return ImageArray(result)

    def copy(self) -> "ImageArray":
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
        if copy:
            scalars = np.array(scalars, dtype=object, copy=True)
        return cls(scalars)

    def __len__(self) -> int:
        return len(self._data)

    def __eq__(self, other) -> np.ndarray:  # type: ignore
        if isinstance(other, ImageArray):
            return np.array([_compare_images(img1, img2) for img1, img2 in zip(self._data, other._data)], dtype=bool)

        if hasattr(other, "__iter__") and not isinstance(other, str):
            if len(other) != len(self):
                return np.repeat(False, len(self))
            return np.array([_compare_images(img1, img2) for img1, img2 in zip(self._data, other)], dtype=bool)
        return np.array([_compare_images(img, other) for img in self._data], dtype=bool)

    @property
    def dtype(self) -> ImageDtype:
        return self._dtype

    @property
    def nbytes(self) -> int:
        return sum(sys.getsizeof(img) for img in self._data if img)

    def __repr__(self) -> str:
        return f"ImageArray([{', '.join([f'<Image: {type(img)}>' if img is not None else 'None' for img in self._data[:5]])}, ...])"

    def _formatter(self, boxed: bool = False):
        return lambda x: f"<Image: {type(x)}>" if x is not None else "None"

    def to_numpy(self, dtype=None, copy=False, na_value=None) -> np.ndarray:
        """Convert the ImageArray to a numpy array."""
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
        """Numpy array interface."""
        return self.to_numpy(dtype=dtype)


def _compare_images(img1, img2) -> bool:
    if img1 is None or img2 is None:
        return img1 is img2

    # Only fetch images when actually comparing
    if isinstance(img1, Image.Image) or isinstance(img2, Image.Image):
        img1 = fetch_image(img1)
        img2 = fetch_image(img2)
        return img1.size == img2.size and img1.mode == img2.mode and img1.tobytes() == img2.tobytes()
    else:
        return img1 == img2
