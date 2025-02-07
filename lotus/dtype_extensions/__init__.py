from lotus.dtype_extensions.image import ImageDtype, ImageArray
from lotus.dtype_extensions.document import DocumentDtype, DocumentArray

import pandas as pd

pd.api.extensions.register_extension_dtype(ImageDtype)
pd.api.extensions.register_extension_dtype(DocumentDtype)


def convert_to_base_data(data: pd.Series | list) -> list:
    """
    Converts data to proper base data type.
    - For original pandas data types, this is returns tolist().
    - For ImageDtype, this returns list of PIL.Image.Image.
    """
    if isinstance(data, pd.Series):
        if isinstance(data.dtype, ImageDtype):
            return [data.array.get_image(i) for i in range(len(data))]
        elif isinstance(data.dtype, DocumentDtype):
            return [data.array.get_document(i) for i in range(len(data))]
        return data.tolist()

    return data


__all__ = ["ImageDtype", "ImageArray", "DocumentDtype", "DocumentArray", "convert_to_base_data"]
