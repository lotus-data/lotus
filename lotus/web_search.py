import os
from enum import Enum

import pandas as pd


class WebSearchCorpus(Enum):
    GOOGLE = "google"
    ARXIV = "arxiv"


def _web_search_google(query: str, K: int, cols: list[str] | None = None) -> pd.DataFrame:
    try:
        from serpapi import GoogleSearch
    except ImportError:
        raise ImportError(
            "The 'serpapi' library is required for Google search. "
            "You can install it with the following command:\n\n"
            "    pip install 'lotus-ai[serpapi]'"
        )
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        raise ValueError("SERPAPI_API_KEY is not set. It is required to run GoogleSearch.")

    search = GoogleSearch(
        {
            "api_key": api_key,
            "q": query,
            "num": K,
        }
    )

    default_cols = [
        "position",
        "title",
        "link",
        "redirect_link",
        "displayed_link",
        "thumbnail",
        "date",
        "author",
        "cited_by",
        "extracted_cited_by",
        "favicon",
        "snippet",
    ]

    results = search.get_dict()
    if "organic_results" not in results:
        raise ValueError("No organic_results found in the response from GoogleSearch")

    df = pd.DataFrame(results["organic_results"])
    columns_to_use = cols if cols is not None else default_cols
    df = df[[col for col in columns_to_use if col in df.columns]]
    return df


def _web_search_arxiv(query: str, K: int, cols: list[str] | None = None) -> pd.DataFrame:
    try:
        import arxiv
    except ImportError:
        raise ImportError(
            "The 'arxiv' library is required for Arxiv search. "
            "You can install it with the following command:\n\n"
            "    pip install 'lotus-ai[arxiv]'"
        )

    client = arxiv.Client()
    search = arxiv.Search(query=query, max_results=K, sort_by=arxiv.SortCriterion.Relevance)
    default_cols = ["id", "title", "link", "abstract", "published", "authors", "categories"]

    articles = []
    for result in client.results(search):
        articles.append(
            {
                "id": result.get_short_id() if hasattr(result, "get_short_id") else result.entry_id,
                "title": result.title,
                "link": result.entry_id,
                "abstract": result.summary,
                "published": result.published,
                "authors": ", ".join([author.name for author in result.authors]),
                "categories": ", ".join(result.categories),
            }
        )
    df = pd.DataFrame(articles)
    columns_to_use = cols if cols is not None else default_cols
    df = df[[col for col in columns_to_use if col in df.columns]]
    return df


def web_search(corpus: WebSearchCorpus, query: str, K: int) -> pd.DataFrame:
    if corpus == WebSearchCorpus.GOOGLE:
        return _web_search_google(query, K)
    elif corpus == WebSearchCorpus.ARXIV:
        return _web_search_arxiv(query, K)
