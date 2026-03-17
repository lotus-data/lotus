from lotus.dtype_extensions.image import ImageDtype, ImageArray
from lotus.dtype_extensions.audio import AudioDtype, AudioArray
import pandas as pd

pd.api.extensions.register_extension_dtype(ImageDtype)
pd.api.extensions.register_extension_dtype(AudioDtype)


def convert_to_base_data(data: pd.Series | list) -> list:
    """
    Converts data to proper base data type.
    - For original pandas data types, this is returns tolist().
    - For ImageDtype, this returns list of PIL.Image.Image.
    - For AudioDtype, this returns list of audio data.
    """
    if isinstance(data, pd.Series):
        if isinstance(data.dtype, ImageDtype):
            return [data.array.get_image(i) for i in range(len(data))]
        if isinstance(data.dtype, AudioDtype):
            return [data.array.get_audio(i) for i in range(len(data))]
        return data.tolist()

    return data


__all__ = ["ImageDtype", "ImageArray", "AudioDtype", "AudioArray", "convert_to_base_data"]

