The Corpus
==========

Overview
--------
A ``Corpus`` is the input to every agentic operator. It normalizes many input forms —
in-memory documents, files, DataFrame rows, or one large text — into a stream of
**units** that can be sharded into bounded batches for parallel agentic processing.

A unit is one atomic segment of the corpus:

.. code-block:: python

    @dataclass
    class Unit:
        id: str                 # stable identifier (e.g. a file path or row index)
        content: str            # the text the agent sees
        metadata: dict          # loader-specific extras (path, row number, chunk index)

Loaders
-------
Build a corpus from whichever form your data takes:

.. code-block:: python

    import lotus

    # In-memory documents (optionally with your own ids)
    lotus.Corpus.from_documents(["doc one", "doc two"], ids=["a", "b"])

    # Files / globs — one unit per file, id = path (great for a codebase)
    lotus.Corpus.from_files("repo/**/*.py")

    # Tabular rows — one unit per row; pick which columns become the content
    lotus.Corpus.from_dataframe(df, content_cols=["title", "body"])

    # One large document, split into fixed-size chunks
    lotus.Corpus.from_text(big_string, chunk_chars=4000)

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Loader
     - Produces
   * - ``from_documents``
     - One unit per string. ``ids`` default to ``0, 1, 2, …``.
   * - ``from_files``
     - One unit per file matching the glob; ``id`` is the path. Recursive by default;
       unreadable files are captured, not fatal.
   * - ``from_dataframe``
     - One unit per row; ``content`` is the selected columns rendered as ``col: value``
       lines (all columns by default).
   * - ``from_text``
     - One unit per ``chunk_chars``-sized chunk of a single text.

Inspecting and sharding
-----------------------
.. code-block:: python

    corpus = lotus.Corpus.from_files("lotus/agentic/*.py")

    len(corpus)              # number of units
    corpus.units             # the list of Unit objects
    corpus.sample(3)         # first 3 units (used by the planner to see the data)
    corpus.shard(2)          # group units into batches of 2 (list of lists)

Sharding controls how the work is divided across parallel agents. In a pipeline the
sharding is chosen by the planner (``shard_size``); ``map`` uses it to batch units per
agent, while ``filter`` always decides per unit.

Running agentic operators
-------------------------
Once you have a corpus, run an agentic pipeline over it with ``corpus.agent``:

.. code-block:: python

    from lotus.tools import PythonREPLTool

    result = corpus.agent(
        task="Summarize each file, then give one architecture overview.",
        ops=["map", "reduce"],
        tools=[PythonREPLTool()],
    )

See :doc:`agentic_operators` for the ops, :doc:`agentic_map_reduce` for the full API, and
:doc:`agentic_filter` for filtering.
