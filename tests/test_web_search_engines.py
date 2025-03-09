import pandas as pd
import pytest

from lotus.web_search import WebSearchCorpus, web_search


class TestWebSearch:
    # @pytest.mark.skipif(not os.getenv("YOU_API_KEY"), reason="Skipping You.com test: API key not set")
    def test_you_search(self):
        df = web_search(WebSearchCorpus.YOU, "AI research", 3)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert any(col in df.columns for col in {"title", "url", "snippet"})

    # @pytest.mark.skipif(not os.getenv("BING_API_KEY"), reason="Skipping Bing test: API key not set")
    def test_bing_search(self):
        df = web_search(WebSearchCorpus.BING, "latest AI models", 3)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert any(col in df.columns for col in {"title", "url", "snippet"})

    # @pytest.mark.skipif(not os.getenv("TAVILY_API_KEY"), reason="Skipping Tavily test: API key not set")
    def test_tavily_search(self):
        df = web_search(WebSearchCorpus.TAVILY, "AI ethics", 3)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert any(col in df.columns for col in {"title", "url", "summary"})

    def test_you_search_fails(self):
        with pytest.raises(ValueError, match="You.com API request failed"):
            web_search(WebSearchCorpus.YOU, "INVALID_QUERY", 3)

    def test_bing_search_fails(self):
        with pytest.raises(ValueError, match="Bing API request failed"):
            web_search(WebSearchCorpus.YOU, "INVALID_QUERY", 3)

    def test_tavily_search_fails(self):
        with pytest.raises(ValueError, match="Tavily API request failed"):
            web_search(WebSearchCorpus.YOU, "INVALID_QUERY", 3)
