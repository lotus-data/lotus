import io
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests  # type: ignore

ALLOWED_METADATA_COLUMNS = [
    "format",
    "title",
    "author",
    "subject",
    "keywords",
    "creator",
    "producer",
    "creationDate",
    "modDate",
    "trapped",
    "encryption",
]


def is_url(path: str | Path) -> bool:
    try:
        result = urlparse(str(path))
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def parse_pdf(
    file_paths: list[str] | str | Path | list[Path],
    per_page: bool = True,
    page_separator: str = "\n",
    metadata_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Parse PDF files and return the content in a DataFrame.

    Args:
        file_paths (list[str] | str | Path | list[Path]): A list of file paths/urls or a single file path/url.
        per_page (bool): If True, return the content of each page as a separate row. Default is True.
        page_separator (str): The separator to use when joining the content of each page in case per_page is False. Default is "\n".
        metadata_columns (list[str] | None): A list of metadata columns to include in the DataFrame. Default is None.
            Allowed metadata columns: ["format", "title", "author", "subject", "keywords", "creator", "producer", "creationDate", "modDate", "trapped", "encryption"].

    Returns:
        pd.DataFrame: A DataFrame with columns: ["file_path", "content", *metadata_columns, "page" (if per_page is True)].
    """
    try:
        import pymupdf
    except ImportError:
        raise ImportError(
            "The 'pymuPDF' library is required for PDF parsing. "
            "You can install it with the following command:\n\n"
            "    pip install 'lotus-ai[pymupdf]'"
        )
    if isinstance(file_paths, str) or isinstance(file_paths, Path):
        file_paths = [file_paths]  # type: ignore
    assert isinstance(file_paths, list), "file_paths must be a list of file paths."

    columns = ["file_path", "content"]
    if metadata_columns:
        for metadata_column in metadata_columns:
            assert (
                metadata_column in ALLOWED_METADATA_COLUMNS
            ), f"{metadata_column} is not an allowed metadata column. Allowed metadata columns: {ALLOWED_METADATA_COLUMNS}"
        columns.extend(metadata_columns)
    else:
        metadata_columns = []

    if per_page:
        columns.append("page")

    all_data = []
    for file_path in file_paths:
        if is_url(file_path):
            response = requests.get(file_path)
            response.raise_for_status()
            opened_doc = pymupdf.open(
                stream=io.BytesIO(response.content), filetype=response.headers.get("Content-Type", "application/pdf")
            )
        else:
            opened_doc = pymupdf.open(file_path)
        data: dict[str, Any] = {
            "file_path": file_path,
        }
        if metadata_columns:
            data.update(
                {
                    metadata_column: opened_doc.metadata.get(metadata_column, None)
                    for metadata_column in metadata_columns
                }
            )
        if per_page:
            data_list: list[dict[str, Any]] = [data.copy() for _ in range(len(opened_doc))]
            for i, page in enumerate(opened_doc):
                data_list[i]["content"] = page.get_text()
                data_list[i]["page"] = i + 1
            all_data.extend(data_list)
        else:
            data["content"] = page_separator.join([page.get_text() for page in opened_doc])
            all_data.append(data)

        opened_doc.close()
    df = pd.DataFrame(all_data, columns=columns)
    return df
