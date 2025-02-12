import pandas as pd


def parse_docs(
    docs: list[str],
    per_page: bool = True,
    header_title: str = "title",
    content_title: str = "content",
) -> pd.DataFrame:
    try:
        import pymupdf
    except ImportError:
        raise ImportError(
            "The 'pymuPDF' library is required for PDF parsing. "
            "You can install it with the following command:\n\n"
            "    pip install 'lotus-ai[pymupdf]'"
        )
    assert header_title != content_title

    df = pd.DataFrame({header_title: [], content_title: []})
    doc_number = 0
    for doc in docs:
        opened_doc = pymupdf.open(doc)
        doc_number += 1

        if per_page:
            page_number = 1
            for page in opened_doc:
                df.loc[-1] = [
                    f"{opened_doc.metadata.get('title', 'DOCUMENT ' + doc_number)} - {opened_doc.metadata.get('author', '')} <{page_number}>",
                    page.get_text().encode("utf8"),
                ]
                df.index += 1
                page_number += 1
        else:
            document_text = [
                page.get_text().encode("utf8") for page in opened_doc
            ].join("\n")
            df.loc[-1] = [
                f"{opened_doc.metadata.get('title', 'DOCUMENT ' + doc_number)} - {opened_doc.metadata.get('author', '')}",
                document_text,
            ]
            df.index += 1

    return df
