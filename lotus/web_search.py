import logging
import os
from enum import Enum

import pandas as pd
import requests  # type: ignore
from dotenv import load_dotenv

load_dotenv()


class WebSearchCorpus(Enum):
    GOOGLE = "google"
    GOOGLE_SCHOLAR = "google_scholar"
    ARXIV = "arxiv"
    YOU = "you"
    BING = "bing"
    TAVILY = "tavily"


def _web_search_google(
    query: str, K: int, cols: list[str] | None = None, engine: str | None = "google"
) -> pd.DataFrame:
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
            "engine": engine,
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
        "inline_links" "publication_info",
        "publication_info",
        "inline_links.cited_by.total",
    ]

    results = search.get_dict()
    if "organic_results" not in results:
        raise ValueError("No organic_results found in the response from GoogleSearch")

    df = pd.DataFrame(results["organic_results"])
    # Unnest nested columns using pandas json_normalize
    if len(df) > 0:  # Only normalize if dataframe is not empty
        df = pd.json_normalize(df.to_dict("records"))
    logging.info("Pruning raw columns: %s", df.columns)
    columns_to_use = cols if cols is not None else default_cols
    # Keep columns that start with any of the default column names
    cols_to_keep = [col for col in df.columns if any(col.startswith(default_col) for default_col in columns_to_use)]
    df = df[cols_to_keep]
    return df


def _web_search_arxiv(query: str, K: int, cols: list[str] | None = None, sort_by_date=False) -> pd.DataFrame:
    try:
        import arxiv
    except ImportError:
        raise ImportError(
            "The 'arxiv' library is required for Arxiv search. "
            "You can install it with the following command:\n\n"
            "    pip install 'lotus-ai[arxiv]'"
        )

    client = arxiv.Client()
    if sort_by_date:
        search = arxiv.Search(query=query, max_results=K, sort_by=arxiv.SortCriterion.SubmittedDate)
    else:
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


def _web_search_you(query: str, K: int, cols: list[str] | None = None) -> pd.DataFrame:
    api_key = os.getenv("YOU_API_KEY")
    if not api_key:
        raise ValueError("YOU_API_KEY is not set. It is required to use You.com search.")

    url = "https://api.ydc-index.io/search"
    params: dict[str, str] = {"q": str(query), "count": str(K)}
    headers = {"X-API-Key": api_key}

    with requests.get(url, headers=headers, params=params) as response:
        response.raise_for_status()

    results = response.json().get("results", [])
    df = pd.DataFrame(results)

    default_cols = ["title", "url", "snippet"]
    columns_to_use = cols if cols is not None else default_cols
    df = df[[col for col in columns_to_use if col in df.columns]]

    return df


def _web_search_bing(query: str, K: int, cols: list[str] | None = None) -> pd.DataFrame:
    api_key = os.getenv("BING_API_KEY")
    if not api_key:
        raise ValueError("BING_API_KEY is not set. It is required to use Bing search.")

    url = "https://api.bing.microsoft.com/v7.0/search"
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    params: dict[str, str] = {"q": str(query), "count": str(K)}

    with requests.get(url, headers=headers, params=params) as response:
        response.raise_for_status()

    results = response.json().get("webPages", {}).get("value", [])
    df = pd.DataFrame(results)

    default_cols = ["name", "url", "snippet"]
    columns_to_use = cols if cols is not None else default_cols
    df = df[[col for col in columns_to_use if col in df.columns]]

    return df


def _web_search_tavily(query: str, K: int, cols: list[str] | None = None) -> pd.DataFrame:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY is not set. It is required to use Tavily search.")

    url = "https://api.tavily.com/search"
    params = {"query": query, "num_results": K, "api_key": api_key}
    headers = {"Authorization": f"Bearer {api_key}"}

    with requests.post(url, headers=headers, json=params) as response:
        response.raise_for_status()

    results = response.json().get("results", [])
    df = pd.DataFrame(results)

    default_cols = ["title", "url", "summary"]
    columns_to_use = cols if cols is not None else default_cols
    df = df[[col for col in columns_to_use if col in df.columns]]

    return df


def web_search(
    corpus: WebSearchCorpus, query: str, K: int, cols: list[str] | None = None, sort_by_date=False
) -> pd.DataFrame:
    if corpus == WebSearchCorpus.GOOGLE:
        return _web_search_google(query, K, cols=cols)
    elif corpus == WebSearchCorpus.ARXIV:
        return _web_search_arxiv(query, K, cols=cols, sort_by_date=sort_by_date)
    elif corpus == WebSearchCorpus.GOOGLE_SCHOLAR:
        return _web_search_google(query, K, engine="google_scholar", cols=cols)
    elif corpus == WebSearchCorpus.YOU:
        return _web_search_you(query, K, cols=cols)
    elif corpus == WebSearchCorpus.BING:
        return _web_search_bing(query, K, cols=cols)
    elif corpus == WebSearchCorpus.TAVILY:
        return _web_search_tavily(query, K, cols=cols)
