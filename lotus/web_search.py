import logging
import os
import time
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


def _fetch_full_text_from_url(urls: list[str], max_length: int | None, delay: float = 0.1) -> list[str | None]:
    results: list[str | None] = []
    for url in urls:
        time.sleep(delay)
        if not url:
            results.append(None)
            continue
        try:
            response = requests.get(url, headers=_DEFAULT_HEADERS, timeout=20)
            response.raise_for_status()
        except Exception as exc:
            logging.warning("Failed to fetch %s: %s", url, exc)
            results.append(None)
            continue

        content_type = response.headers.get("Content-Type", "").lower()
        if "application/pdf" in content_type or url.lower().endswith(".pdf"):
            pdf_text = _extract_text_from_pdf(response.content, url, max_length)
            if pdf_text:
                results.append(pdf_text)
                continue

        html_text = response.text or response.content.decode("utf-8", errors="ignore")
        text = _extract_text_from_html(html_text)
        results.append(_truncate_text(text, max_length) if text else None)
    return results


def _web_extract_tavily(urls: list[str], max_length: int | None) -> list[str | None]:
    """
    Extract content from URLs using Tavily Extract API (supports batch).

    Args:
        urls: URLs to extract content from.
        max_length: Optional maximum character length per extracted text.

    Returns:
        List of extracted texts in the same order as *urls* (None on failure).
    """
    if not urls:
        return []

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY is not set. It is required to use Tavily extract.")

    url_endpoint = "https://api.tavily.com/extract"
    headers = {"Authorization": f"Bearer {api_key}"}

    payload: dict[str, str | list[str]] = {
        "urls": urls,
        "extract_depth": "basic",
        "format": "markdown",
    }

    try:
        with requests.post(url_endpoint, headers=headers, json=payload) as response:
            response.raise_for_status()
            response_data = response.json()
    except Exception as exc:
        logging.warning("Failed to extract content from %s using Tavily: %s", urls, exc)
        return [None] * len(urls)

    result_map: dict[str, str | None] = {}
    for r in response_data.get("results", []):
        r_url = r.get("url", "")
        raw_content = r.get("raw_content")
        if raw_content and isinstance(raw_content, str):
            result_map[r_url] = _truncate_text(raw_content, max_length)
        else:
            result_map[r_url] = None

    for f in response_data.get("failed_results", []):
        f_url = f.get("url", "")
        logging.warning("Tavily extraction failed for URL %s: %s", f_url, f.get("error", "Unknown error"))
        result_map[f_url] = None

    return [result_map.get(u) for u in urls]


def _extract_full_text_for_identifiers(
    corpus: WebSearchCorpus,
    identifiers: list[str],
    max_length: int | None,
    delay: float = 0.1,
) -> list[str | None]:
    if corpus == WebSearchCorpus.ARXIV:
        try:
            import arxiv
        except ImportError:
            raise ImportError(
                "The 'arxiv' library is required for Arxiv search. "
                "You can install it with the following command:\n\n"
                "    pip install 'lotus-ai[arxiv]'"
            )
        pdf_urls = [f"https://arxiv.org/pdf/{ident}.pdf" if ident else "" for ident in identifiers]
        pdf_texts = _fetch_full_text_from_url(pdf_urls, max_length, delay)
        results: list[str | None] = []
        for ident, pdf_text in zip(identifiers, pdf_texts):
            if pdf_text:
                results.append(pdf_text)
            elif ident:
                article = next(arxiv.Client().results(arxiv.Search(id_list=[ident], max_results=1)), None)
                results.append(article.summary if article else None)
            else:
                results.append(None)
        return results

    elif corpus == WebSearchCorpus.PUBMED:
        try:
            from pymed import PubMed
        except ImportError:
            raise ImportError(
                "The 'pymed' library is required for PubMed search. "
                "You can install it with the following command:\n\n"
                "    pip install 'lotus-ai[pubmed]'"
            )
        pm = PubMed(tool="LOTUS")
        results = []
        for ident in identifiers:
            if not ident:
                results.append(None)
                continue
            article = next(pm.query(ident, max_results=1), None)
            results.append(article.abstract if article else None)
        return results

    elif corpus == WebSearchCorpus.TAVILY:
        valid = [ident for ident in identifiers if ident]
        texts = _web_extract_tavily(valid, max_length)
        text_map = dict(zip(valid, texts))
        return [text_map.get(ident) if ident else None for ident in identifiers]  # type: ignore[arg-type]

    valid_urls = [ident or "" for ident in identifiers]
    return _fetch_full_text_from_url(valid_urls, max_length, delay)


def _web_search_google(
    queries: list[str],
    K: int,
    cols: list[str] | None = None,
    engine: str | None = "google",
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    delay: float = 0.1,
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

    columns_to_use = cols if cols is not None else _GOOGLE_DEFAULT_COLS
    dfs: list[pd.DataFrame] = []
    for query in queries:
        time.sleep(delay)
        orig_query = query

        search_params: dict[str, str | int | None] = {
            "api_key": api_key,
            "q": query,
            "num": K,
            "engine": engine,
        }

        if start_date or end_date:
            if start_date and end_date:
                tbs_value = f"cdr:1,cd_min:{start_date.strftime('%m/%d/%Y')},cd_max:{end_date.strftime('%m/%d/%Y')}"
                search_params["tbs"] = tbs_value
                search_params["q"] = orig_query
            elif start_date:
                search_params["q"] = f"{orig_query} after:{start_date.strftime('%Y-%m-%d')}"
            elif end_date:
                search_params["q"] = f"{orig_query} before:{end_date.strftime('%Y-%m-%d')}"

        search = GoogleSearch(search_params)
        results = search.get_dict()
        if "organic_results" not in results:
            raise ValueError("No organic_results found in the response from GoogleSearch")

        df = pd.DataFrame(results["organic_results"])
        if len(df) > 0:
            df = pd.json_normalize(df.to_dict("records"))
        logging.info("Pruning raw columns: %s", df.columns)
        df = df[columns_to_use]
        df["query"] = query
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def _web_search_arxiv(
    queries: list[str],
    K: int,
    cols: list[str] | None = None,
    sort_by_date: bool = False,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    delay: float = 0.1,
) -> pd.DataFrame:
    try:
        import arxiv
    except ImportError:
        raise ImportError(
            "The 'arxiv' library is required for Arxiv search. "
            "You can install it with the following command:\n\n"
            "    pip install 'lotus-ai[arxiv]'"
        )

    columns_to_use = cols if cols is not None else _ARXIV_DEFAULT_COLS
    client = arxiv.Client()
    dfs: list[pd.DataFrame] = []

    for query in queries:
        time.sleep(delay)
        search_query = query
        if start_date or end_date:
            if start_date and end_date:
                date_filter = (
                    f"submittedDate:[{start_date.strftime('%Y%m%d%H%M')} TO {end_date.strftime('%Y%m%d%H%M')}]"
                )
                search_query = f"({query}) AND ({date_filter})"
            elif start_date:
                date_filter = f"submittedDate:[{start_date.strftime('%Y%m%d%H%M')} TO 99999999]"
                search_query = f"({query}) AND ({date_filter})"
            elif end_date:
                date_filter = f"submittedDate:[00000000 TO {end_date.strftime('%Y%m%d%H%M')}]"
                search_query = f"({query}) AND ({date_filter})"

        sort_criterion = arxiv.SortCriterion.SubmittedDate if sort_by_date else arxiv.SortCriterion.Relevance
        search = arxiv.Search(query=search_query, max_results=K, sort_by=sort_criterion)

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
        df = df[[col for col in columns_to_use if col in df.columns]]
        df["query"] = query
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def _web_search_you(
    queries: list[str],
    K: int,
    cols: list[str] | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    delay: float = 0.1,
) -> pd.DataFrame:
    api_key = os.getenv("YOU_API_KEY")
    if not api_key:
        raise ValueError("YOU_API_KEY is not set. It is required to use You.com search.")

    columns_to_use = cols if cols is not None else _YOU_DEFAULT_COLS
    dfs: list[pd.DataFrame] = []

    for query in queries:
        time.sleep(delay)
        url = "https://ydc-index.io/v1/search"
        params: dict[str, str | int] = {"query": str(query), "count": K}
        headers = {"X-API-Key": api_key}

        if start_date or end_date:
            if start_date and end_date:
                freshness = f"{start_date.strftime('%Y-%m-%d')}to{end_date.strftime('%Y-%m-%d')}"
                params["freshness"] = freshness
            elif start_date:
                today = datetime.now()
                freshness = f"{start_date.strftime('%Y-%m-%d')}to{today.strftime('%Y-%m-%d')}"
                params["freshness"] = freshness
            elif end_date:
                freshness = f"0000-01-01to{end_date.strftime('%Y-%m-%d')}"
                params["freshness"] = freshness

        with requests.get(url, headers=headers, params=params) as response:
            response.raise_for_status()

        response_data = response.json()
        results: list[dict] = []
        if "results" in response_data:
            results_data = response_data["results"]
            if "web" in results_data:
                results.extend(results_data["web"])
            if "news" in results_data:
                results.extend(results_data["news"])

        df = pd.DataFrame(results)
        df = df[[col for col in columns_to_use if col in df.columns]]
        df["query"] = query
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def _web_search_bing(
    queries: list[str],
    K: int,
    cols: list[str] | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    delay: float = 0.1,
) -> pd.DataFrame:
    raise DeprecationWarning("Bing search is discontinued. Please use Google search instead.")


def _web_search_tavily(
    queries: list[str],
    K: int,
    cols: list[str] | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    delay: float = 0.1,
) -> pd.DataFrame:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY is not set. It is required to use Tavily search.")

    columns_to_use = cols if cols is not None else _TAVILY_DEFAULT_COLS
    headers = {"Authorization": f"Bearer {api_key}"}
    dfs: list[pd.DataFrame] = []

    for query in queries:
        time.sleep(delay)
        url = "https://api.tavily.com/search"
        params: dict[str, str | int] = {"query": query, "max_results": K}

        if start_date:
            params["start_date"] = start_date.strftime("%Y-%m-%d")
        if end_date:
            params["end_date"] = end_date.strftime("%Y-%m-%d")

        with requests.post(url, headers=headers, json=params) as response:
            response.raise_for_status()

        results = response.json().get("results", [])
        df = pd.DataFrame(results)
        df = df[[col for col in columns_to_use if col in df.columns]]
        df["query"] = query
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def _web_search_pubmed(
    queries: list[str],
    K: int,
    cols: list[str] | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    delay: float = 0.1,
) -> pd.DataFrame:
    try:
        from pymed import PubMed
    except ImportError:
        raise ImportError(
            "The 'pymed' library is required for PubMed search. "
            "You can install it with the following command:\n\n"
            "    pip install 'lotus-ai[pubmed]'"
        )

    tool = os.getenv("PUBMED_TOOL", "LOTUS")
    columns_to_use = cols if cols is not None else _PUBMED_DEFAULT_COLS
    pubmed = PubMed(tool=tool)
    dfs: list[pd.DataFrame] = []

    for query in queries:
        time.sleep(delay)
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

        results = pubmed.query(search_query, max_results=K)

        articles = []
        for article in results:
            authors_str = ""
            if hasattr(article, "authors") and article.authors:
                authors_str = ", ".join(
                    [
                        f"{author.get('firstname', '')} {author.get('lastname', '')}".strip()
                        for author in article.authors
                    ]
                )

            pmid = None
            if hasattr(article, "pubmed_id"):
                pubmed_id_value = article.pubmed_id
                if isinstance(pubmed_id_value, dict):
                    pubmed_id_value = pubmed_id_value.get("pubmed_id", "")
                if isinstance(pubmed_id_value, str):
                    pmid = pubmed_id_value.split("\n")[0].strip() if pubmed_id_value else None
                elif pubmed_id_value:
                    pmid = str(pubmed_id_value)

            link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}" if pmid else None

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
        df = df[[col for col in columns_to_use if col in df.columns]]
        df["query"] = query
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def web_search(
    corpus: WebSearchCorpus,
    query: str | list[str],
    K: int,
    cols: list[str] | None = None,
    sort_by_date: bool = False,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    delay: float = 0.1,
) -> pd.DataFrame:
    """
    Perform web search across different search engines.

    Args:
        corpus: The search engine to use (GOOGLE, GOOGLE_SCHOLAR, ARXIV, YOU, BING, TAVILY, PUBMED)
        query: The search query string, or a list of query strings.
        K: Maximum number of results to return per query.
        cols: Optional list of columns to include in the results.
        sort_by_date: Whether to sort results by date (currently only supported for ARXIV).
        start_date: Optional start date for filtering results (as a datetime object).
                   Supported for GOOGLE, GOOGLE_SCHOLAR, ARXIV, TAVILY, and YOU.
        end_date: Optional end date for filtering results (as a datetime object).
                 Supported for GOOGLE, GOOGLE_SCHOLAR, ARXIV, TAVILY, and YOU.

    Returns:
        A pandas DataFrame containing the search results with a ``query`` column.

    Raises:
        ValueError: If date format is invalid or required API keys are not set
    """
    queries = [query] if isinstance(query, str) else query
    if corpus == WebSearchCorpus.GOOGLE:
        return _web_search_google(queries, K, cols=cols, start_date=start_date, end_date=end_date, delay=delay)
    elif corpus == WebSearchCorpus.ARXIV:
        return _web_search_arxiv(
            queries, K, cols=cols, sort_by_date=sort_by_date, start_date=start_date, end_date=end_date, delay=delay
        )
    elif corpus == WebSearchCorpus.GOOGLE_SCHOLAR:
        return _web_search_google(
            queries, K, engine="google_scholar", cols=cols, start_date=start_date, end_date=end_date, delay=delay
        )
    elif corpus == WebSearchCorpus.YOU:
        return _web_search_you(queries, K, cols=cols, start_date=start_date, end_date=end_date, delay=delay)
    elif corpus == WebSearchCorpus.BING:
        return _web_search_bing(queries, K, cols=cols, delay=delay)
    elif corpus == WebSearchCorpus.TAVILY:
        return _web_search_tavily(queries, K, cols=cols, start_date=start_date, end_date=end_date, delay=delay)
    elif corpus == WebSearchCorpus.PUBMED:
        return _web_search_pubmed(queries, K, cols=cols, start_date=start_date, end_date=end_date, delay=delay)
    else:
        raise ValueError(f"Unsupported corpus: {corpus}")


def _get_url_from_id(corpus: WebSearchCorpus, doc_id: str) -> str:
    if corpus == WebSearchCorpus.ARXIV:
        return f"https://arxiv.org/abs/{doc_id}"
    elif corpus == WebSearchCorpus.PUBMED:
        return f"https://pubmed.ncbi.nlm.nih.gov/{doc_id}/"
    else:
        return doc_id


def _get_id_from_url(corpus: WebSearchCorpus, url: str) -> str:
    if corpus == WebSearchCorpus.ARXIV:
        return url.split("/")[4]
    elif corpus == WebSearchCorpus.PUBMED:
        return url.split("/")[4]
    else:
        return url


def web_extract(
    corpus: WebSearchCorpus,
    doc_ids: str | list[str] | None = None,
    urls: str | list[str] | None = None,
    max_length: int | None = None,
    delay: float = 0.1,
) -> pd.DataFrame:
    """
    Extract full text from specific ids/urls across different search engines.

    Accepts a single value or a list of values for ``doc_ids`` / ``urls``.
    When the underlying API supports batching (e.g. Tavily), a single
    request is made; otherwise each identifier is fetched individually.

    Args:
        corpus: The search engine to use (GOOGLE, GOOGLE_SCHOLAR, ARXIV, YOU, BING, TAVILY, PUBMED)
        doc_ids: Corpus-specific identifier(s). Required for ARXIV/PUBMED when url is not provided.
        urls: URL(s) to fetch. For non-ARXIV/PUBMED corpora, doc_ids is treated as urls.
        max_length: Optional maximum character length for extracted full text.

    Returns:
        A pandas DataFrame with columns: id, url, and full_text.
    """
    if corpus == WebSearchCorpus.BING:
        raise DeprecationWarning("Bing search is discontinued. Please use Google search instead.")

    # Normalise to lists
    doc_id_list: list[str]
    if isinstance(doc_ids, str):
        doc_id_list = [doc_ids.strip()]
    elif isinstance(doc_ids, list):
        doc_id_list = [d.strip() for d in doc_ids]
    else:
        doc_id_list = []

    url_list: list[str]
    if isinstance(urls, str):
        url_list = [urls.strip()]
    elif isinstance(urls, list):
        url_list = [u.strip() for u in urls]
    else:
        url_list = []

    if not url_list and not doc_id_list:
        raise ValueError("web_extract requires doc_id or url.")

    identifiers = doc_id_list + [_get_id_from_url(corpus, u) for u in url_list]
    row_urls = [_get_url_from_id(corpus, d) for d in doc_id_list] + url_list
    texts = _extract_full_text_for_identifiers(corpus, identifiers, max_length, delay)
    return pd.DataFrame({"id": identifiers, "url": row_urls, "full_text": texts})
