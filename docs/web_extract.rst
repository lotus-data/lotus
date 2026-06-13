web_extract
============

``web_extract`` extracts full text from URLs or corpus-specific document IDs
and returns the results as a pandas DataFrame. Use it after :doc:`web_search`
when search results point to documents you want to process, or use it directly
when you already know the document IDs or URLs.

Basic Extraction
----------------

``doc_ids`` and ``urls`` each accept either a string or a list of strings. The
result has ``id``, ``url``, and ``full_text`` columns.

.. code-block:: python

    from lotus import WebSearchCorpus, web_extract

    df = web_extract(
        WebSearchCorpus.ARXIV,
        doc_ids="2303.08774",
    )

    print(df[["id", "url", "full_text"]])

Extract Multiple Documents
--------------------------

.. code-block:: python

    df = web_extract(
        WebSearchCorpus.TAVILY,
        urls=[
            "https://en.wikipedia.org/wiki/Artificial_intelligence",
            "https://en.wikipedia.org/wiki/Machine_learning",
        ],
        max_length=20_000,
    )

When the provider supports batching, LOTUS sends one batched request.
Otherwise it fetches each identifier separately. ``delay`` controls the pause
between non-batched fetches.

Document IDs and URLs
---------------------

For arXiv and PubMed, ``doc_ids`` are converted to canonical document URLs.
For other corpora, ``doc_ids`` are treated as URLs. Passing ``urls`` always
uses the given URL directly.

.. code-block:: python

    pubmed = web_extract(
        WebSearchCorpus.PUBMED,
        doc_ids=["12345678", "23456789"],
    )

    page = web_extract(
        WebSearchCorpus.YOU,
        urls="https://example.com/article",
    )

Using Extracted Text
--------------------

The returned DataFrame works with semantic operators and LazyFrames. For
example, you can extract papers and then summarize their full text.

.. code-block:: python

    papers = web_extract(
        WebSearchCorpus.ARXIV,
        doc_ids=["2407.11418", "2309.06180"],
        max_length=40_000,
    )

    summary = papers.sem_agg(
        "Summarize the shared technical themes across {full_text}."
    )

Parameters
----------

.. code-block:: python

    web_extract(
        corpus,
        doc_ids=None,
        urls=None,
        max_length=None,
        delay=0.1,
    )

API Reference
-------------

.. autofunction:: lotus.web_search.web_extract
