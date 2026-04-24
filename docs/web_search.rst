web_search
===========

``web_search`` loads web search results into a pandas DataFrame. Use it when
you need a tabular set of search results before applying semantic operators,
pandas transformations, or a LazyFrame pipeline.

Use :doc:`web_extract` when you already have URLs or corpus-specific document
IDs and want the full text.

Supported corpora are:

- ``WebSearchCorpus.GOOGLE``
- ``WebSearchCorpus.GOOGLE_SCHOLAR``
- ``WebSearchCorpus.ARXIV``
- ``WebSearchCorpus.YOU``
- ``WebSearchCorpus.TAVILY``
- ``WebSearchCorpus.PUBMED``
- ``WebSearchCorpus.BING``; Bing is discontinued and raises a deprecation
  warning in the current implementation.

Basic Search
------------

``web_search`` accepts one query or a list of queries and returns one DataFrame
with a ``query`` column.

.. code-block:: python

    from lotus import WebSearchCorpus, web_search

    df = web_search(
        WebSearchCorpus.ARXIV,
        query="lazy dataframe query optimization",
        K=5,
    )

    print(df[["title", "abstract", "query"]])

Search Multiple Queries
-----------------------

.. code-block:: python

    df = web_search(
        WebSearchCorpus.PUBMED,
        query=[
            "large language models clinical summarization",
            "retrieval augmented generation medicine",
        ],
        K=3,
    )

Date Filtering
--------------

``start_date`` and ``end_date`` filter results for Google, Google Scholar,
arXiv, You.com, Tavily, and PubMed. ``sort_by_date`` is supported for arXiv.

.. code-block:: python

    from datetime import datetime
    from lotus import WebSearchCorpus, web_search

    df = web_search(
        WebSearchCorpus.ARXIV,
        "transformer architecture",
        10,
        sort_by_date=True,
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 31),
    )

Select Columns
--------------

Use ``cols`` to request a subset of result fields.

.. code-block:: python

    df = web_search(
        WebSearchCorpus.TAVILY,
        "AI safety evaluations",
        5,
        cols=["title", "url", "content"],
    )

Common default columns include:

- arXiv: ``id``, ``title``, ``link``, ``abstract``, ``published``,
  ``authors``, ``categories``
- Google and Google Scholar: ``title``, ``link``, ``snippet``, ``date``,
  ``publication_info``
- You.com: ``title``, ``url``, ``snippets``, ``description``
- Tavily: ``title``, ``url``, ``content``
- PubMed: ``id``, ``title``, ``link``, ``abstract``, ``published``,
  ``authors``, ``journal``, ``doi``, ``methods``, ``results``,
  ``conclusions``

Required Setup
--------------

- Google and Google Scholar require ``SERPAPI_API_KEY`` and the ``serpapi``
  extra.
- arXiv requires the ``arxiv`` extra.
- PubMed requires the ``pubmed`` extra.
- You.com requires ``YOU_API_KEY`` and the ``web_search`` extra.
- Tavily requires ``TAVILY_API_KEY`` and the ``web_search`` extra.

.. code-block:: console

    $ pip install "lotus-ai[serpapi]"
    $ pip install "lotus-ai[arxiv]"
    $ pip install "lotus-ai[pubmed]"
    $ pip install "lotus-ai[web_search]"

Parameters
----------

.. code-block:: python

    web_search(
        corpus,
        query,
        K,
        cols=None,
        sort_by_date=False,
        start_date=None,
        end_date=None,
        delay=0.1,
    )

API Reference
-------------

.. autoclass:: lotus.web_search.WebSearchCorpus
   :members:

.. autofunction:: lotus.web_search.web_search
