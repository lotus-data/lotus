import logging
import os
from datetime import datetime
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
    PUBMED = "pubmed"


def _web_search_google(
    query: str,
    K: int,
    cols: list[str] | None = None,
    engine: str | None = "google",
    start_date: datetime | None = None,
    end_date: datetime | None = None,
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

    orig_query = query  # keep unnaffected version

    search_params: dict[str, str | int | None] = {
        "api_key": api_key,
        "q": query,
        "num": K,
        "engine": engine,
    }

    if start_date or end_date:
        if start_date and end_date:
            # Format dates as MM/DD/YYYY for tbs parameter
            tbs_value = f"cdr:1,cd_min:{start_date.strftime('%m/%d/%Y')},cd_max:{end_date.strftime('%m/%d/%Y')}"
            search_params["tbs"] = tbs_value
            # for search query itself, use original
            search_params["q"] = orig_query
        elif start_date:
            # Only start date - use after: operator in query, format is YYYY-MM-DD for after:
            search_params["q"] = f"{orig_query} after:{start_date.strftime('%Y-%m-%d')}"
        elif end_date:
            # Only end date - use before: operator in query, format is YYYY-MM-DD for before:
            search_params["q"] = f"{orig_query} before:{end_date.strftime('%Y-%m-%d')}"

    search = GoogleSearch(search_params)

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
        "inline_links",
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
    df = df[columns_to_use]
    return df


def _web_search_arxiv(
    query: str,
    K: int,
    cols: list[str] | None = None,
    sort_by_date=False,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> pd.DataFrame:
    try:
        import arxiv
    except ImportError:
        raise ImportError(
            "The 'arxiv' library is required for Arxiv search. "
            "You can install it with the following command:\n\n"
            "    pip install 'lotus-ai[arxiv]'"
        )

    # Add date filtering to query if dates are provided
    search_query = query
    if start_date or end_date:
        if start_date and end_date:
            # Format dates as YYYYMMDD for arXiv
            date_filter = f"submittedDate:[{start_date.strftime('%Y%m%d%H%M')} TO {end_date.strftime('%Y%m%d%H%M')}]"
            search_query = f"({query}) AND ({date_filter})"
        elif start_date:
            date_filter = f"submittedDate:[{start_date.strftime('%Y%m%d%H%M')} TO 99999999]"
            search_query = f"({query}) AND ({date_filter})"
        elif end_date:
            date_filter = f"submittedDate:[00000000 TO {end_date.strftime('%Y%m%d%H%M')}]"
            search_query = f"({query}) AND ({date_filter})"

    client = arxiv.Client()
    if sort_by_date:
        search = arxiv.Search(query=search_query, max_results=K, sort_by=arxiv.SortCriterion.SubmittedDate)
    else:
        search = arxiv.Search(query=search_query, max_results=K, sort_by=arxiv.SortCriterion.Relevance)

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


def _web_search_you(
    query: str,
    K: int,
    cols: list[str] | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> pd.DataFrame:
    api_key = os.getenv("YOU_API_KEY")
    if not api_key:
        raise ValueError("YOU_API_KEY is not set. It is required to use You.com search.")

    url = "https://ydc-index.io/v1/search"
    params: dict[str, str | int] = {"query": str(query), "count": K}
    headers = {"X-API-Key": api_key}

    # Add freshness parameter for date filtering
    # According to docs: freshness can be "day", "week", "month", "year", or "YYYY-MM-DDtoYYYY-MM-DD"
    if start_date or end_date:
        if start_date and end_date:
            # Use date range format: YYYY-MM-DDtoYYYY-MM-DD (no space between dates)
            freshness = f"{start_date.strftime('%Y-%m-%d')}to{end_date.strftime('%Y-%m-%d')}"
            params["freshness"] = freshness
        elif start_date:
            # Only start date - calculate approximate range to today
            # For simplicity, use date range from start_date to today
            today = datetime.now()
            freshness = f"{start_date.strftime('%Y-%m-%d')}to{today.strftime('%Y-%m-%d')}"
            params["freshness"] = freshness
        elif end_date:
            # Only end date - use date range from a very early date to end_date
            # Using a reasonable early date (e.g., 2000-01-01)
            freshness = f"0000-01-01to{end_date.strftime('%Y-%m-%d')}"
            params["freshness"] = freshness

    with requests.get(url, headers=headers, params=params) as response:
        response.raise_for_status()

    response_data = response.json()
    # You.com API returns results in a structure with 'web' and 'news' arrays
    # Flatten both into a single list
    results = []
    if "results" in response_data:
        results_data = response_data["results"]
        if "web" in results_data:
            results.extend(results_data["web"])
        if "news" in results_data:
            results.extend(results_data["news"])

    df = pd.DataFrame(results)

    default_cols = ["title", "url", "snippets", "description"]
    columns_to_use = cols if cols is not None else default_cols
    df = df[[col for col in columns_to_use if col in df.columns]]

    return df


def _web_search_bing(
    query: str,
    K: int,
    cols: list[str] | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> pd.DataFrame:
    raise DeprecationWarning("Bing search is discontinued. Please use Google search instead.")


def _web_search_tavily(
    query: str,
    K: int,
    cols: list[str] | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> pd.DataFrame:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY is not set. It is required to use Tavily search.")

    url = "https://api.tavily.com/search"
    params: dict[str, str | int] = {"query": query, "max_results": K}

    # Add date filtering if provided
    # Tavily API supports start_date and end_date parameters in YYYY-MM-DD format
    if start_date:
        params["start_date"] = start_date.strftime("%Y-%m-%d")
    if end_date:
        params["end_date"] = end_date.strftime("%Y-%m-%d")

    headers = {"Authorization": f"Bearer {api_key}"}

    with requests.post(url, headers=headers, json=params) as response:
        response.raise_for_status()

    results = response.json().get("results", [])
    df = pd.DataFrame(results)

    default_cols = ["title", "url", "content"]
    columns_to_use = cols if cols is not None else default_cols
    df = df[[col for col in columns_to_use if col in df.columns]]

    return df


def _web_search_pubmed(
    query: str,
    K: int,
    cols: list[str] | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> pd.DataFrame:
    try:
        from pymed import PubMed
    except ImportError:
        raise ImportError(
            "The 'pymed' library is required for PubMed search. "
            "You can install it with the following command:\n\n"
            "    pip install 'lotus-ai[pubmed]'"
        )

    # Get tool and email from environment variables or use defaults
    tool = os.getenv("PUBMED_TOOL", "LOTUS")
    email = os.getenv("PUBMED_EMAIL")
    if not email:
        raise ValueError(
            "PUBMED_EMAIL environment variable is not set. "
            "It is required to use PubMed search. Please set it to your email address."
        )

    # Add date filtering to query if dates are provided
    # PubMed query syntax: dates can be added as "YYYY:YYYY[PDAT]" for publication dates
    search_query = query
    if start_date or end_date:
        if start_date and end_date:
            date_filter = f"{start_date.year}:{end_date.year}[PDAT]"
            search_query = f"({query}) AND {date_filter}"
        elif start_date:
            date_filter = f"{start_date.year}:3000[PDAT]"
            search_query = f"({query}) AND {date_filter}"
        elif end_date:
            date_filter = f"1800:{end_date.year}[PDAT]"
            search_query = f"({query}) AND {date_filter}"

    pubmed = PubMed(tool=tool, email=email)
    results = pubmed.query(search_query, max_results=K)

    default_cols = ["title", "link", "abstract", "publication_date", "authors", "pmid"]

    articles = []
    for article in results:
        # Extract authors as comma-separated string
        authors_str = ""
        if article.authors:
            authors_str = ", ".join([f"{author.get('firstname', '')} {author.get('lastname', '')}".strip() for author in article.authors])

        # Create PubMed link
        pmid = article.pubmed_id if hasattr(article, "pubmed_id") else None
        link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}" if pmid else None

        articles.append(
            {
                "title": article.title if hasattr(article, "title") else None,
                "link": link,
                "abstract": article.abstract if hasattr(article, "abstract") else None,
                "publication_date": article.publication_date if hasattr(article, "publication_date") else None,
                "authors": authors_str,
                "pmid": pmid,
            }
        )

    df = pd.DataFrame(articles)
    columns_to_use = cols if cols is not None else default_cols
    df = df[[col for col in columns_to_use if col in df.columns]]
    return df


def web_search(
    corpus: WebSearchCorpus,
    query: str,
    K: int,
    cols: list[str] | None = None,
    sort_by_date=False,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> pd.DataFrame:
    """
    Perform web search across different search engines.

    Args:
        corpus: The search engine to use (GOOGLE, GOOGLE_SCHOLAR, ARXIV, YOU, BING, TAVILY, PUBMED)
        query: The search query string
        K: Maximum number of results to return
        cols: Optional list of columns to include in the results
        sort_by_date: Whether to sort results by date (currently only supported for ARXIV)
        start_date: Optional start date for filtering results (as a datetime object).
                   Supported for GOOGLE, GOOGLE_SCHOLAR, ARXIV, TAVILY, YOU, and PUBMED.
        end_date: Optional end date for filtering results (as a datetime object).
                 Supported for GOOGLE, GOOGLE_SCHOLAR, ARXIV, TAVILY, YOU, and PUBMED.

    Returns:
        A pandas DataFrame containing the search results

    Raises:
        ValueError: If date format is invalid or required API keys are not set
    """
    if corpus == WebSearchCorpus.GOOGLE:
        return _web_search_google(query, K, cols=cols, start_date=start_date, end_date=end_date)
    elif corpus == WebSearchCorpus.ARXIV:
        return _web_search_arxiv(
            query, K, cols=cols, sort_by_date=sort_by_date, start_date=start_date, end_date=end_date
        )
    elif corpus == WebSearchCorpus.GOOGLE_SCHOLAR:
        return _web_search_google(
            query, K, engine="google_scholar", cols=cols, start_date=start_date, end_date=end_date
        )
    elif corpus == WebSearchCorpus.YOU:
        return _web_search_you(query, K, cols=cols, start_date=start_date, end_date=end_date)
    elif corpus == WebSearchCorpus.BING:
        return _web_search_bing(query, K, cols=cols)
    elif corpus == WebSearchCorpus.TAVILY:
        return _web_search_tavily(query, K, cols=cols, start_date=start_date, end_date=end_date)
    elif corpus == WebSearchCorpus.PUBMED:
        return _web_search_pubmed(query, K, cols=cols, start_date=start_date, end_date=end_date)
