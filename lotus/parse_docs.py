import pandas as pd

ALLOWED_METADATA_COLUMNS = ['format', 'title', 'author', 'subject', 'keywords', 'creator', 'producer', 'creationDate', 'modDate', 'trapped', 'encryption']
    
    
def parse_pdf(
    file_paths: list[str] | str,
    per_page: bool = True,
    page_separator: str = "\n",
    metadata_columns: list[str] | None = None,
) -> pd.DataFrame:
    try:
        import pymupdf
    except ImportError:
        raise ImportError(
            "The 'pymuPDF' library is required for PDF parsing. "
            "You can install it with the following command:\n\n"
            "    pip install 'lotus-ai[pymupdf]'"
        )
    if not isinstance(file_paths, list) and not isinstance(file_paths, tuple):
        file_paths = [file_paths]

    columns = ['file_path', 'content']
    if metadata_columns:
        for metadata_column in metadata_columns:
            assert metadata_column in ALLOWED_METADATA_COLUMNS, f"{metadata_column} is not an allowed metadata column. Allowed metadata columns: {ALLOWED_METADATA_COLUMNS}"
        columns.extend(metadata_columns)
    else:
        metadata_columns = []

    if per_page:
        columns.append('page')

    all_data = []
    for file_path in file_paths:
        opened_doc = pymupdf.open(file_path)
        data = {
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
            data_list = [data.copy() for _ in range(len(opened_doc))]
            for i, page in enumerate(opened_doc):
                data_list[i]["content"] = page.get_text()
                data_list[i]["page"] = i + 1
            all_data.extend(data_list)
        else:
            data["content"] = page_separator.join(
                [page.get_text() for page in opened_doc]
            )
            all_data.append(data)
        
        opened_doc.close()
    df = pd.DataFrame(all_data, columns=columns)
    return df
