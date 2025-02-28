web_search
========================

Overview
---------
The `web_search` function allows you to search the web for information, then process that data with LOTUS.
Different search engines are supported, including Google, Google Scholar and Arxiv.

Arxiv Example
--------
.. code-block:: python

    import lotus
    from lotus import WebSearchCorpus, web_search
    from lotus.models import LM

    lm = LM(model="gpt-4o-mini")

    lotus.settings.configure(lm=lm)

    df = web_search(WebSearchCorpus.ARXIV, "deep learning", 5)[["title", "abstract"]]
    print(f"Results from Arxiv\n{df}\n\n")

    most_interesting_articles = df.sem_topk("Which {abstract} is most exciting?", K=1)
    print(f"Most interesting article: \n{most_interesting_articles.iloc[0]}")

Google Example
--------
Before running the following example, you need to set the `SERPAPI_API_KEY` environment variable.

.. code-block:: python

    import lotus
    from lotus import WebSearchCorpus, web_search
    from lotus.models import LM

    lm = LM(model="gpt-4o-mini")

    lotus.settings.configure(lm=lm)

    df = web_search(WebSearchCorpus.GOOGLE, "deep learning research", 5)[["title", "snippet"]]
    print(f"Results from Google\n{df}")
    most_interesting_articles = df.sem_topk("Which {snippet} is the most exciting?", K=1)
    print(f"Most interesting articles\n{most_interesting_articles}")

Required Parameters
--------------------
- **corpus** : The search corpus to use. Available options:
  - ``WebSearchCorpus.ARXIV``: Search academic papers on arxiv.org
  - ``WebSearchCorpus.GOOGLE``: Search the web using Google Search
  - ``WebSearchCorpus.GOOGLE_SCHOLAR``: Search academic papers using Google Scholar
- **query** : The query to search for
- **K** : The number of results to return

Optional Parameters
--------------------
- **cols** : The columns to take from the API search results. Default values should be sufficient for most use cases. To see available columns, enable logging:

  .. code-block:: python

      import logging
      logging.basicConfig(level=logging.INFO)

