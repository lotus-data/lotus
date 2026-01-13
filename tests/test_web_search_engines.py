from datetime import datetime, timezone

import pandas as pd

from lotus.web_search import WebSearchCorpus, web_search


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
