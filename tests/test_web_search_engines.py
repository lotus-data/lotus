import pandas as pd

from lotus.web_search import WebSearchCorpus, web_search


class TestWebSearch:
    def test_you_search(self):
        df = web_search(WebSearchCorpus.YOU, "AI research", 3)
        print(df)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert all(col in df.columns for col in {"title", "url", "snippet"})

    def test_bing_search(self):
        df = web_search(WebSearchCorpus.BING, "latest AI models", 3)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert all(col in df.columns for col in {"title", "url", "snippet"})

    def test_tavily_search(self):
        df = web_search(WebSearchCorpus.TAVILY, "AI ethics", 3)
        print(df)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert all(col in df.columns for col in {"title", "url", "summary"})
