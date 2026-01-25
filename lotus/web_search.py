import logging
import os
from datetime import datetime
from enum import Enum
from html.parser import HTMLParser

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


_GOOGLE_DEFAULT_COLS = [
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
_ARXIV_DEFAULT_COLS = ["id", "title", "link", "abstract", "published", "authors", "categories"]
_YOU_DEFAULT_COLS = ["title", "url", "snippets", "description"]
_TAVILY_DEFAULT_COLS = ["title", "url", "content"]
_PUBMED_DEFAULT_COLS = [
    "id",
    "title",
    "link",
    "abstract",
    "published",
    "authors",
    "categories",
    "journal",
    "doi",
    "methods",
    "conclusions",
    "results",
]

_DEFAULT_COLS_BY_CORPUS = {
    WebSearchCorpus.GOOGLE: _GOOGLE_DEFAULT_COLS,
    WebSearchCorpus.GOOGLE_SCHOLAR: _GOOGLE_DEFAULT_COLS,
    WebSearchCorpus.ARXIV: _ARXIV_DEFAULT_COLS,
    WebSearchCorpus.YOU: _YOU_DEFAULT_COLS,
    WebSearchCorpus.TAVILY: _TAVILY_DEFAULT_COLS,
    WebSearchCorpus.PUBMED: _PUBMED_DEFAULT_COLS,
}

_DEFAULT_HEADERS = {
    "User-Agent": "lotus-ai/1.1 (+https://github.com/lotus-ai/lotus)",
}


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style", "noscript"):
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "noscript") and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        text = data.strip()
        if text:
            self._chunks.append(text)

    def get_text(self) -> str:
        return " ".join(self._chunks)


def _truncate_text(text: str | None, max_length: int | None) -> str | None:
    if text is None:
        return None
    if max_length is None:
        return text
    if max_length <= 0:
        return ""
    if len(text) <= max_length:
        return text
    return text[:max_length]


def _extract_text_from_html(html_text: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(html_text)
    parser.close()
    return parser.get_text()


def _extract_text_from_pdf(pdf_bytes: bytes, url: str, max_length: int | None) -> str | None:
    try:
        import fitz  # type: ignore
    except ImportError:
        raise ImportError(
            "The 'pymupdf' library is required for PDF extraction. "
            "You can install it with the following command:\n\n"
            "    pip install 'lotus-ai[file_extractor]'"
        )

    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            parts: list[str] = []
            total_len = 0
            for page in doc:
                page_text = page.get_text("text")
                if page_text:
                    if max_length is not None and max_length > 0:
                        remaining = max_length - total_len
                        if remaining <= 0:
                            break
                        if len(page_text) > remaining:
                            page_text = page_text[:remaining]
                    parts.append(page_text)
                    total_len += len(page_text)
                if max_length is not None and total_len >= max_length:
                    break
        text = "\n".join(parts).strip()
        return text or None
    except Exception as exc:
        logging.warning("Failed to extract PDF text from %s: %s", url, exc)
        return None


def _fetch_full_text_from_url(url: str, max_length: int | None) -> str | None:
    if not url:
        return None
    try:
        response = requests.get(url, headers=_DEFAULT_HEADERS, timeout=20)
        response.raise_for_status()
    except Exception as exc:
        logging.warning("Failed to fetch %s: %s", url, exc)
        return None

    content_type = response.headers.get("Content-Type", "").lower()
    if "application/pdf" in content_type or url.lower().endswith(".pdf"):
        pdf_text = _extract_text_from_pdf(response.content, url, max_length)
        if pdf_text:
            return pdf_text

    html_text = response.text or response.content.decode("utf-8", errors="ignore")
    text = _extract_text_from_html(html_text)
    if not text:
        return None
    return _truncate_text(text, max_length)


def _web_extract_tavily(url: str, max_length: int | None) -> str | None:
    """
    Extract content from a URL using Tavily Extract API.

    Args:
        url: URL to extract content from
        max_length: Optional maximum character length for extracted text

    Returns:
        Extracted text content, or None if extraction failed
    """
    if not url:
        return None

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY is not set. It is required to use Tavily extract.")

    url_endpoint = "https://api.tavily.com/extract"
    headers = {"Authorization": f"Bearer {api_key}"}

    # Build request payload with default settings
    payload: dict[str, str | list[str]] = {
        "urls": [url],
        "extract_depth": "basic",
        "format": "markdown",
    }

    try:
        with requests.post(url_endpoint, headers=headers, json=payload) as response:
            response.raise_for_status()
            response_data = response.json()
    except Exception as exc:
        logging.warning("Failed to extract content from %s using Tavily: %s", url, exc)
        return None

    # Extract raw_content from results
    results = response_data.get("results", [])
    if not results:
        # Check for failed_results
        failed_results = response_data.get("failed_results", [])
        if failed_results:
            error_msg = failed_results[0].get("error", "Unknown error")
            logging.warning("Tavily extraction failed for URL %s: %s", url, error_msg)
        return None

    # Get raw_content from first result
    raw_content = results[0].get("raw_content")
    if not raw_content or not isinstance(raw_content, str):
        return None

    # Truncate if max_length is provided
    return _truncate_text(raw_content, max_length)


def _extract_full_text_for_identifier(
    corpus: WebSearchCorpus,
    identifier: str | None,
    url: str | None,
    max_length: int | None,
) -> str | None:
    if not identifier:
        return None

    if corpus == WebSearchCorpus.ARXIV:
        text = _fetch_full_text_from_url(f"https://arxiv.org/pdf/{identifier}.pdf", max_length)
        if text:
            return text
        try:
            import arxiv
        except ImportError:
            raise ImportError(
                "The 'arxiv' library is required for Arxiv search. "
                "You can install it with the following command:\n\n"
                "    pip install 'lotus-ai[arxiv]'"
            )
        article = next(arxiv.Client().results(arxiv.Search(id_list=[identifier], max_results=1)), None)
        if article:
            return article.summary
        return None

    elif corpus == WebSearchCorpus.PUBMED:
        try:
            from pymed import PubMed
        except ImportError:
            raise ImportError(
                "The 'pymed' library is required for PubMed search. "
                "You can install it with the following command:\n\n"
                "    pip install 'lotus-ai[pubmed]'"
            )
        article = next(PubMed(tool="LOTUS").query(identifier, max_results=1), None)
        if article:
            return article.abstract
        return None

    elif corpus == WebSearchCorpus.TAVILY:
        return _web_extract_tavily(identifier, max_length)

    return _fetch_full_text_from_url(identifier, max_length)


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

    results = search.get_dict()
    if "organic_results" not in results:
        raise ValueError("No organic_results found in the response from GoogleSearch")

    df = pd.DataFrame(results["organic_results"])
    # Unnest nested columns using pandas json_normalize
    if len(df) > 0:  # Only normalize if dataframe is not empty
        df = pd.json_normalize(df.to_dict("records"))
    logging.info("Pruning raw columns: %s", df.columns)
    columns_to_use = cols if cols is not None else _GOOGLE_DEFAULT_COLS
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
    columns_to_use = cols if cols is not None else _ARXIV_DEFAULT_COLS
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

    columns_to_use = cols if cols is not None else _YOU_DEFAULT_COLS
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

    columns_to_use = cols if cols is not None else _TAVILY_DEFAULT_COLS
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

    # Get tool from environment variable or use default
    tool = os.getenv("PUBMED_TOOL", "LOTUS")

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

    pubmed = PubMed(tool=tool)
    results = pubmed.query(search_query, max_results=K)

    articles = []
    for article in results:
        authors_str = ""
        if hasattr(article, "authors") and article.authors:
            authors_str = ", ".join(
                [f"{author.get('firstname', '')} {author.get('lastname', '')}".strip() for author in article.authors]
            )

        # PMID parsing, handle dict/newline cases and use first ID
        pmid = None
        if hasattr(article, "pubmed_id"):
            pubmed_id_value = article.pubmed_id
            # Handle if it's a dict
            if isinstance(pubmed_id_value, dict):
                pubmed_id_value = pubmed_id_value.get("pubmed_id", "")
            # Handle if it's a string with newlines
            if isinstance(pubmed_id_value, str):
                pmid = pubmed_id_value.split("\n")[0].strip() if pubmed_id_value else None
            elif pubmed_id_value:
                pmid = str(pubmed_id_value)

        # Create PubMed link
        link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}" if pmid else None

        # Extract categories
        categories_str = ""
        if hasattr(article, "publication_types") and article.publication_types:
            categories_str = ", ".join([pt.get("name", "") for pt in article.publication_types if pt.get("name")])

        articles.append(
            {
                "id": pmid,
                "title": article.title if hasattr(article, "title") else None,
                "link": link,
                "abstract": article.abstract if hasattr(article, "abstract") else None,
                "published": article.publication_date if hasattr(article, "publication_date") else None,
                "authors": authors_str,
                "categories": categories_str,
                "journal": article.journal if hasattr(article, "journal") else None,
                "doi": article.doi if hasattr(article, "doi") else None,
                "methods": article.methods if hasattr(article, "methods") else None,
                "conclusions": article.conclusions if hasattr(article, "conclusions") else None,
                "results": article.results if hasattr(article, "results") else None,
            }
        )

    df = pd.DataFrame(articles)
    columns_to_use = cols if cols is not None else _PUBMED_DEFAULT_COLS
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
        corpus: The search engine to use (GOOGLE, GOOGLE_SCHOLAR, ARXIV, YOU, BING, TAVILY)
        query: The search query string
        K: Maximum number of results to return
        cols: Optional list of columns to include in the results
        sort_by_date: Whether to sort results by date (currently only supported for ARXIV)
        start_date: Optional start date for filtering results (as a datetime object).
                   Supported for GOOGLE, GOOGLE_SCHOLAR, ARXIV, TAVILY, and YOU.
        end_date: Optional end date for filtering results (as a datetime object).
                 Supported for GOOGLE, GOOGLE_SCHOLAR, ARXIV, TAVILY, and YOU.

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


def web_extract(
    corpus: WebSearchCorpus,
    doc_id: str | None = None,
    url: str | None = None,
    max_length: int | None = None,
) -> pd.DataFrame:
    """
    Extract full text from specific ids/urls across different search engines.

    Args:
        corpus: The search engine to use (GOOGLE, GOOGLE_SCHOLAR, ARXIV, YOU, BING, TAVILY, PUBMED)
        doc_id: Optional corpus-specific identifier (required for ARXIV/PUBMED if url not provided).
        url: Optional URL to fetch. For non-ARXIV/PUBMED corpora, doc_id is treated as url.
        max_length: Optional maximum character length for extracted full text.

    Returns:
        A pandas DataFrame with columns: id, url, and full_text.
    """
    if corpus == WebSearchCorpus.BING:
        raise DeprecationWarning("Bing search is discontinued. Please use Google search instead.")

    doc_id = doc_id.strip() if doc_id else None
    url = url.strip() if url else None
    if not doc_id and not url:
        raise ValueError("web_extract requires doc_id or url.")

    if doc_id and url:
        raise ValueError("web_extract requires doc_id or url, but not both.")

    # Build simple DataFrame with id, url, and full_text
    data: dict[str, str | None] = {"id": None, "url": None, "full_text": None}

    if corpus in (WebSearchCorpus.ARXIV, WebSearchCorpus.PUBMED):
        # For ARXIV/PUBMED, use doc_id as id and construct url if needed
        if doc_id:
            data["id"] = doc_id
            if corpus == WebSearchCorpus.ARXIV:
                data["url"] = f"https://arxiv.org/abs/{doc_id}"
            else:
                data["url"] = f"https://pubmed.ncbi.nlm.nih.gov/{doc_id}/"
        else:
            data["url"] = url
            assert url is not None
            if corpus == WebSearchCorpus.ARXIV:
                data["id"] = url.split("/")[4]
            else:
                data["id"] = url.split("/")[4]
    else:
        # For other corpora, use identifier as url
        data["url"] = data["id"] = doc_id or url

    # Extract full text
    data["full_text"] = _extract_full_text_for_identifier(corpus, data["id"], data["url"], max_length)

    return pd.DataFrame([data])
