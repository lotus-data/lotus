web_search
========================

Overview
---------
The `web_search` function allows you to load documents from the web, then process that data with LOTUS.
Different search engines are supported, including Google, Google Scholar, Arxiv, You.com, Bing and Tavily.

Arxiv Example
--------
To get started, you will need to install the lotus submodule as follows:
.. code-block:: shell
    pip install lotus[arxiv]

Then you can run your lotus program:
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
Before running the following example, you need to set the `SERPAPI_API_KEY` environment variable. You will also need to install the lotus submodule as follows:
.. code-block:: shell
    pip install lotus[serpapi]

Then you can run your lotus program:

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

You.com Example
--------
Before running the following example, you need to set the `YOU_API_KEY` environment variable. You will also need to install the lotus submodule as follows:
.. code-block:: shell
    pip install lotus[you]

Then you can run your lotus program:

.. code-block:: python

    import lotus
    from lotus import WebSearchCorpus, web_search
    from lotus.models import LM

    lm = LM(model="gpt-4o-mini")

    lotus.settings.configure(lm=lm)

    df = web_search(WebSearchCorpus.YOU, "latest AI breakthroughs", 10)[["title", "snippet"]]
    print(f"Results from You.com:\n{df}\n")
    top_you_articles = df.sem_topk("Which {snippet} is the most groundbreaking?", K=3)
    print(f"Top 3 most interesting articles from You.com:\n{top_you_articles}")


Bing Example
--------
Before running the following example, you need to set the `BING_API_KEY` environment variable. You will also need to install the lotus submodule as follows:
.. code-block:: shell
    pip install lotus[bing]

Then you can run your lotus program:

.. code-block:: python

    import lotus
    from lotus import WebSearchCorpus, web_search
    from lotus.models import LM

    lm = LM(model="gpt-4o-mini")

    lotus.settings.configure(lm=lm)

    df = web_search(WebSearchCorpus.BING, "state-of-the-art AI models", 10)[["title", "snippet"]]
    print(f"Results from Bing:\n{df}\n")
    top_bing_articles = df.sem_topk("Which {snippet} provides the best insight into AI models?", K=3)
    print(f"Top 3 most insightful articles from Bing:\n{top_bing_articles}")


Tavily Example
--------
Before running the following example, you need to set the `TAVILY_API_KEY` environment variable. You will also need to install the lotus submodule as follows:
.. code-block:: shell
    pip install lotus[tavily]

Then you can run your lotus program:

.. code-block:: python

    import lotus
    from lotus import WebSearchCorpus, web_search
    from lotus.models import LM

    lm = LM(model="gpt-4o-mini")

    lotus.settings.configure(lm=lm)

    df = web_search(WebSearchCorpus.TAVILY, "AI ethics in 2025", 10)[["title", "summary"]]
    print(f"Results from Tavily:\n{df}\n")
    top_tavily_articles = df.sem_topk("Which {summary} best explains ethical concerns in AI?", K=3)
    print(f"Top 3 articles from Tavily on AI ethics:\n{top_tavily_articles}")


Required Parameters
--------------------
- **corpus** : The search corpus to use. Available options:
  - ``WebSearchCorpus.ARXIV``: Search academic papers on arxiv.org
  - ``WebSearchCorpus.GOOGLE``: Search the web using Google Search
  - ``WebSearchCorpus.GOOGLE_SCHOLAR``: Search academic papers using Google Scholar
  - ``WebSearchCorpus.YOU``: Search the web using You.com
  - ``WebSearchCorpus.BING``: Search the web using Bing
  - ``WebSearchCorpus.TAVILY``: Search the web using Tavily
- **query** : The query to search for
- **K** : The number of results to return

Optional Parameters
--------------------
- **cols** : The columns to take from the API search results. Default values should be sufficient for most use cases. To see available columns, enable logging:

  .. code-block:: python

      import logging
      logging.basicConfig(level=logging.INFO)

