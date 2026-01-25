web_search
========================

Overview
---------
The `web_search` function allows you to load documents from the web, then process that data with LOTUS.

Different search engines are supported, including Google, Google Scholar, Arxiv, You.com and Tavily.

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


Date Filtering Example
--------------------
You can filter search results by date range using the ``start_date`` and ``end_date`` parameters:

.. code-block:: python

    from datetime import datetime
    from lotus import WebSearchCorpus, web_search

    # Search for papers published in 2024
    start = datetime(2024, 1, 1)
    end = datetime(2024, 12, 31)
    
    df = web_search(
        WebSearchCorpus.ARXIV, 
        "transformer architecture", 
        10,
        start_date=start,
        end_date=end
    )
    
    # Search for recent news from the past month
    from datetime import timedelta
    one_month_ago = datetime.now() - timedelta(days=30)
    
    df = web_search(
        WebSearchCorpus.TAVILY,
        "AI developments",
        10,
        start_date=one_month_ago
    )


Required Parameters
--------------------
- **corpus** : The search corpus to use. Available options:
  - ``WebSearchCorpus.ARXIV``: Search academic papers on arxiv.org
  - ``WebSearchCorpus.GOOGLE``: Search the web using Google Search
  - ``WebSearchCorpus.GOOGLE_SCHOLAR``: Search academic papers using Google Scholar
  - ``WebSearchCorpus.YOU``: Search the web using You.com
  - ``WebSearchCorpus.TAVILY``: Search the web using Tavily
- **query** : The query to search for
- **K** : The number of results to return

Optional Parameters
--------------------
- **cols** : The columns to take from the API search results. Default values should be sufficient for most use cases. To see available columns, enable logging:

  .. code-block:: python

      import logging
      logging.basicConfig(level=logging.INFO)

- **start_date** : Optional start date for filtering results (as a ``datetime`` object). 
  Returns only results created or published on or after this date. 

- **end_date** : Optional end date for filtering results (as a ``datetime`` object). 
  Returns only results created or published on or before this date. 


web_extract
========================

Overview
---------
The `web_extract` function allows you to extract full text content from specific URLs or document IDs across different search engines. This is useful when you already know the URL or ID of a document and want to extract its full content for processing with LOTUS.

The function returns a simple DataFrame with three columns: ``id``, ``url``, and ``full_text``.

Arxiv Extract Example
--------
To get started, you will need to install the lotus submodule as follows:
.. code-block:: shell
    pip install lotus[arxiv]

Then you can run your lotus program:

.. code-block:: python

    import lotus
    from lotus import WebSearchCorpus, web_extract
    from lotus.models import LM

    lm = LM(model="gpt-4o-mini")

    lotus.settings.configure(lm=lm)

    # Extract full text from an arXiv paper using its ID
    df = web_extract(WebSearchCorpus.ARXIV, doc_id="2303.08774")
    print(f"Extracted from ArXiv:\n{df}\n\n")

    # Use the extracted full text for semantic operations
    if df["full_text"].iloc[0]:
        print(f"Full text length: {len(df['full_text'].iloc[0])} characters")


Tavily Extract Example
--------
Before running the following example, you need to set the `TAVILY_API_KEY` environment variable. You will also need to install the lotus submodule as follows:
.. code-block:: shell
    pip install lotus[tavily]

Then you can run your lotus program:

.. code-block:: python

    import lotus
    from lotus import WebSearchCorpus, web_extract
    from lotus.models import LM

    lm = LM(model="gpt-4o-mini")

    lotus.settings.configure(lm=lm)

    # Extract full text from a URL using Tavily Extract API
    df = web_extract(WebSearchCorpus.TAVILY, url="https://en.wikipedia.org/wiki/Artificial_intelligence")
    print(f"Extracted from Tavily:\n{df}\n\n")

    # Use the extracted full text for semantic operations
    if df["full_text"].iloc[0]:
        print(f"Full text length: {len(df['full_text'].iloc[0])} characters")


PubMed Extract Example
--------
To get started, you will need to install the lotus submodule as follows:
.. code-block:: shell
    pip install lotus[pubmed]

Then you can run your lotus program:

.. code-block:: python

    import lotus
    from lotus import WebSearchCorpus, web_extract
    from lotus.models import LM

    lm = LM(model="gpt-4o-mini")

    lotus.settings.configure(lm=lm)

    # Extract full text from a PubMed article using its ID
    df = web_extract(WebSearchCorpus.PUBMED, doc_id="12345678")
    print(f"Extracted from PubMed:\n{df}\n\n")

    # Use the extracted full text for semantic operations
    if df["full_text"].iloc[0]:
        print(f"Full text length: {len(df['full_text'].iloc[0])} characters")

web_extract Required Parameters
--------------------
- **corpus** : The search corpus to use. Available options:
  - ``WebSearchCorpus.ARXIV``: Extract from academic papers on arxiv.org
  - ``WebSearchCorpus.GOOGLE``: Extract from URLs using standard HTTP fetching
  - ``WebSearchCorpus.GOOGLE_SCHOLAR``: Extract from URLs using standard HTTP fetching
  - ``WebSearchCorpus.YOU``: Extract from URLs using standard HTTP fetching
  - ``WebSearchCorpus.TAVILY``: Extract from URLs using Tavily Extract API
  - ``WebSearchCorpus.PUBMED``: Extract from PubMed articles
- **doc_id** or **url** : Either a corpus-specific identifier (required for ARXIV/PUBMED if url not provided) or a URL to fetch. You must provide exactly one of these parameters.

web_extract Optional Parameters
--------------------
- **max_length** : Optional maximum character length for extracted full text. If provided, the extracted text will be truncated to this length.
