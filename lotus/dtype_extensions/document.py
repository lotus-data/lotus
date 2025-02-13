import sys
from typing import Sequence, Union

import numpy as np
import pandas as pd
import pymupdf
from pandas.api.extensions import ExtensionArray, ExtensionDtype
from pymupdf import Document

from lotus.utils import fetch_document


class DocumentDtype(ExtensionDtype):
    name = "document"
    type = Document
    na_value = None

    @classmethod
    def construct_array_type(cls):
        return DocumentArray


class DocumentArray(ExtensionArray):
    def __init__(self, values):
        self._data = np.asarray(values, dtype=object)
        self._dtype = DocumentDtype()
        self.allowed_document_types = ["Document", "string"]
        self._cached_documents: dict[tuple[int, str], str | Document | None] = {}  # Cache for loaded documents

    def __getitem__(self, item: int | slice | Sequence[int]) -> np.ndarray:
        result = self._data[item]

        if isinstance(item, (int, np.integer)):
            # Return the raw value for display purposes
            return result

        return DocumentArray(result)

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
        for doc_type in self.allowed_document_types:
            if (idx, doc_type) in self._cached_documents:
                del self._cached_documents[(idx, doc_type)]

    def get_document(self, idx: int, doc_type: str = "Document") -> Union[Document, str, None]:
        """Explicit method to fetch and return the actual document"""
        if (idx, doc_type) not in self._cached_documents:
            document_result = fetch_document(self._data[idx], doc_type)
            assert document_result is None or isinstance(document_result, (Document, str))
            self._cached_documents[(idx, doc_type)] = document_result
        return self._cached_documents[(idx, doc_type)]

    def isna(self) -> np.ndarray:
        return pd.isna(self._data)

    def take(self, indices: Sequence[int], allow_fill: bool = False, fill_value=None) -> "DocumentArray":
        result = self._data.take(indices, axis=0)
        if allow_fill and fill_value is not None:
            result[indices == -1] = fill_value
        return DocumentArray(result)

    def copy(self) -> "DocumentArray":
        new_array = DocumentArray(self._data.copy())
        new_array._cached_documents = self._cached_documents.copy()
        return new_array

    def _concat_same_type(cls, to_concat: Sequence["DocumentArray"]) -> "DocumentArray":
        """
        Concatenate multiple DocumentArray instances into a single one.

        Args:
            to_concat (Sequence[DocumentArray]): A sequence of DocumentArray instances to concatenate.

        Returns:
            DocumentArray: A new DocumentArray containing all elements from the input arrays.
        """
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
        if isinstance(other, DocumentArray):
            return np.array(
                [_compare_documents(doc1, doc2) for doc1, doc2 in zip(self._data, other._data)],
                dtype=bool,
            )

        if hasattr(other, "__iter__") and not isinstance(other, str):
            if len(other) != len(self):
                return np.repeat(False, len(self))
            return np.array(
                [_compare_documents(doc1, doc2) for doc1, doc2 in zip(self._data, other)],
                dtype=bool,
            )
        return np.array([_compare_documents(doc, other) for doc in self._data], dtype=bool)

    @property
    def dtype(self) -> DocumentDtype:
        return self._dtype

    @property
    def nbytes(self) -> int:
        return sum(sys.getsizeof(doc) for doc in self._data if doc)

    def __repr__(self) -> str:
        return f"DocumentArray([{', '.join([f'<Document: {doc.metadata}>' if isinstance(doc, Document) else f'<Document: {doc}>' for doc in self._data[:5]])}, ...])"

    def _formatter(self, boxed: bool = False):
        return lambda x: (f"<Document: {x.metadata}>" if isinstance(x, Document) else f"<Document: {x}")

    def to_numpy(self, dtype=None, copy=False, na_value=None) -> np.ndarray:
        """Convert the DocumentArray to a numpy array."""
        documents = []
        for i, doc_data in enumerate(self._data):
            if isinstance(doc_data, Document):
                text = doc_data.metadata.__str__() + "\n"
                for page in doc_data.pages:
                    text += page.get_text() + "\n"
                documents.append(text)
            elif isinstance(doc_data, str):
                doc = pymupdf.open(doc_data)
                text = doc.metadata.__str__() + "\n"
                for page in doc:
                    text += page.get_text() + "\n"
                documents.append(text)
        result = np.empty(len(self), dtype=object)
        result[:] = documents
        return result

    def __array__(self, dtype=None) -> np.ndarray:
        """Numpy array interface."""
        return self.to_numpy(dtype=dtype)


def _compare_documents(doc1, doc2) -> bool:
    if doc1 is None or doc2 is None:
        return doc1 is doc2

    # Only fetch documents when actually comparing
    if isinstance(doc1, Document) and isinstance(doc2, Document):
        if doc1.page_count == doc2.page_count:
            for doc1_page, doc2_page in zip(doc1.pages, doc2.pages):
                if doc1_page.extract_text() != doc2_page.extract_text():
                    return False
        return doc1.metadata == doc2.metadata
    else:
        return doc1 == doc2
