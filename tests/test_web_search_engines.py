from datetime import datetime, timezone

import pandas as pd

from lotus.web_search import WebSearchCorpus, web_extract, web_search


class TestWebSearch:
    def test_you_search(self):
        df = web_search(WebSearchCorpus.YOU, "AI research", 3)
        print(df)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert all(col in df.columns for col in {"title", "url", "snippet"})

    def test_arxiv_search(self):
        df = web_search(WebSearchCorpus.ARXIV, "latest AI models", 3)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert all(col in df.columns for col in {"title", "link", "abstract", "published", "authors", "categories"})

    def test_tavily_search(self):
        df = web_search(WebSearchCorpus.TAVILY, "AI ethics", 3)
        print(df)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert all(col in df.columns for col in {"title", "url", "content"})

    def test_start_end_date_filtering_google(self):
        df = web_search(
            WebSearchCorpus.GOOGLE,
            "AI ethics",
            3,
            cols=["title", "link", "snippet", "date"],
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 1, 31),
        )
        print(df)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert all(col in df.columns for col in {"title", "link", "snippet", "date"})

        df["parsed_date"] = pd.to_datetime(df["date"], format="%b %d, %Y")

        assert df["parsed_date"].min() >= datetime(2025, 1, 1)
        assert df["parsed_date"].max() <= datetime(2025, 1, 31)

    def test_start_end_date_filtering_arxiv(self):
        df = web_search(
            WebSearchCorpus.ARXIV,
            "AI ethics",
            3,
            cols=["title", "link", "abstract", "published"],
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 1, 31),
        )
        print(df)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert all(col in df.columns for col in {"title", "link", "abstract", "published"})
        assert df["published"].min() >= datetime(2025, 1, 1, 0, 0, 0, 0, tzinfo=timezone.utc)
        assert df["published"].max() <= datetime(2025, 1, 31, 0, 0, 0, 0, tzinfo=timezone.utc)

    def test_pubmed_search(self):
        df = web_search(WebSearchCorpus.PUBMED, "machine learning", 3)
        print(df)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert all(
            col in df.columns for col in {"id", "title", "link", "abstract", "published", "authors", "categories"}
        )


class TestWebExtract:
    def test_arxiv_extract_by_id(self):
        df = web_extract(WebSearchCorpus.ARXIV, doc_id="2303.08774")
        print(df)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert all(col in df.columns for col in {"id", "url", "full_text"})
        assert df["id"].iloc[0] == "2303.08774"
        assert "arxiv.org" in df["url"].iloc[0]
        # full_text may be None if extraction fails, but should be present in structure
        assert "full_text" in df.columns

    def test_arxiv_extract_by_url(self):
        df = web_extract(WebSearchCorpus.ARXIV, url="https://arxiv.org/abs/2303.08774")
        print(df)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert all(col in df.columns for col in {"id", "url", "full_text"})
        assert df["url"].iloc[0] == "https://arxiv.org/abs/2303.08774"

    def test_tavily_extract(self):
        df = web_extract(WebSearchCorpus.TAVILY, url="https://en.wikipedia.org/wiki/Artificial_intelligence")
        print(df)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert all(col in df.columns for col in {"id", "url", "full_text"})
        assert df["url"].iloc[0] == "https://en.wikipedia.org/wiki/Artificial_intelligence"

    def test_tavily_extract_with_max_length(self):
        df = web_extract(
            WebSearchCorpus.TAVILY,
            url="https://en.wikipedia.org/wiki/Machine_learning",
            max_length=1000,
        )
        print(df)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        if df["full_text"].iloc[0]:
            assert len(df["full_text"].iloc[0]) <= 1000

    def test_pubmed_extract_by_id(self):
        # Using a known PubMed ID
        df = web_extract(WebSearchCorpus.PUBMED, doc_id="12345678")
        print(df)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert all(col in df.columns for col in {"id", "url", "full_text"})
        assert df["id"].iloc[0] == "12345678"
        assert "pubmed.ncbi.nlm.nih.gov" in df["url"].iloc[0]

    def test_pubmed_extract_by_url(self):
        df = web_extract(WebSearchCorpus.PUBMED, url="https://pubmed.ncbi.nlm.nih.gov/12345678/")
        print(df)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert all(col in df.columns for col in {"id", "url", "full_text"})

    def test_you_extract(self):
        df = web_extract(WebSearchCorpus.YOU, url="https://en.wikipedia.org/wiki/Deep_learning")
        print(df)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert all(col in df.columns for col in {"id", "url", "full_text"})
        assert df["url"].iloc[0] == "https://en.wikipedia.org/wiki/Deep_learning"

    def test_extract_requires_doc_id_or_url(self):
        try:
            web_extract(WebSearchCorpus.TAVILY)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "requires doc_id or url" in str(e)

    def test_extract_requires_only_one(self):
        try:
            web_extract(WebSearchCorpus.TAVILY, doc_id="test", url="https://example.com")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "but not both" in str(e)
