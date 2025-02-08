import os
from enum import Enum

import arxiv
import pandas as pd
from serpapi import GoogleSearch


class ExternalSearchCorpus(Enum):
    GOOGLE = "google"
    ARXIV = "arxiv"


def _sem_external_search_google(query: str, K: int) -> pd.DataFrame:
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
    cols = [
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
    df = df[[col for col in cols if col in df.columns]]
    return df


def _sem_external_search_arxiv(query: str, K: int) -> pd.DataFrame:
    client = arxiv.Client()
    search = arxiv.Search(query=query, max_results=K, sort_by=arxiv.SortCriterion.Relevance)
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
    return df


def sem_external_search(corpus: ExternalSearchCorpus, query: str, K: int) -> pd.DataFrame:
    if corpus == ExternalSearchCorpus.GOOGLE:
        return _sem_external_search_google(query, K)
    elif corpus == ExternalSearchCorpus.ARXIV:
        return _sem_external_search_arxiv(query, K)
