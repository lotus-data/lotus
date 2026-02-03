AST and Lazy Evaluation Framework
==================================

The ``lotus.ast`` module provides a lazy evaluation framework for building and executing semantic operator pipelines. This framework allows you to construct complex data processing pipelines declaratively and execute them efficiently with automatic caching and optimization.

Overview
--------

The AST (Abstract Syntax Tree) module implements a lazy evaluation system where operations are not executed immediately but are instead stored as a sequence of nodes. This allows for:

- **Lazy Execution**: Operations are deferred until explicitly executed
- **Automatic Caching**: Intermediate results are cached based on content-addressable keys
- **Pipeline Optimization**: Built-in optimizers can transform pipelines for better performance
- **Composability**: Pipelines can reference other pipelines, enabling complex data flows

Core Components
---------------

LazyFrame
~~~~~~~~~

The ``LazyFrame`` class is the main interface for building lazy pipelines. It wraps a pandas DataFrame with a sequence of operations that are executed on demand.

Creating a LazyFrame
^^^^^^^^^^^^^^^^^^^^

You can create a ``LazyFrame`` in several ways:

.. code-block:: python

    from lotus.ast import LazyFrame
    import pandas as pd

    # Create a LazyFrame with a source key
    lazy_df = LazyFrame("queries")

    # Create a LazyFrame with a bound DataFrame
    df = pd.DataFrame({"text": ["Hello", "World"]})
    lazy_df = LazyFrame("queries", df=df)

The source key is used to identify which input DataFrame to use when executing the pipeline.

Building Pipelines
^^^^^^^^^^^^^^^^^^

LazyFrame supports all semantic operators and pandas operations:

.. code-block:: python

    # Build a pipeline with semantic operators
    pipeline = (
        LazyFrame("data")
        .sem_filter("text contains positive sentiment")
        .sem_map("extract the main topic from {text}")
        .sem_topk("most relevant topics", K=5)
    )

    # Mix semantic and pandas operations
    pipeline = (
        LazyFrame("data")
        .filter(lambda df: df["score"] > 0.5)  # Pandas filter
        .sem_filter("text is relevant")        # Semantic filter
        .assign(processed=lambda df: df["text"].str.upper())  # Pandas assign
    )

Executing Pipelines
^^^^^^^^^^^^^^^^^^^

Execute a pipeline using the ``execute()`` method:

.. code-block:: python

    # Single DataFrame input
    result = pipeline.execute(input_df)

    # Multiple DataFrame inputs (for joins)
    result = pipeline.execute({
        "left": left_df,
        "right": right_df
    })

The ``run()`` method creates a ``LazyFrameRun`` object that provides more control over execution:

.. code-block:: python

    run = pipeline.run(input_df)
    result = run.execute()
    cache_stats = run.cache_stats  # Get cache hit/miss statistics

Inspecting Pipelines
^^^^^^^^^^^^^^^^^^^^

View the pipeline structure using the ``show()`` method:

.. code-block:: python

    print(pipeline.show())

This displays a tree representation of all nodes in the pipeline.

Nodes
~~~~~

Nodes are the building blocks of pipelines. Each node represents a single operation that transforms a DataFrame.

SourceNode
^^^^^^^^^^

The ``SourceNode`` represents the input data source:

.. code-block:: python

    from lotus.ast import SourceNode

    source = SourceNode(key="data", df=my_dataframe)

Semantic Operator Nodes
^^^^^^^^^^^^^^^^^^^^^^^

All semantic operators have corresponding node types:

- ``SemFilterNode``: Semantic filtering
- ``SemMapNode``: Semantic mapping
- ``SemExtractNode``: Semantic extraction
- ``SemAggNode``: Semantic aggregation
- ``SemTopKNode``: Semantic top-k selection
- ``SemJoinNode``: Semantic join
- ``SemSimJoinNode``: Semantic similarity join
- ``SemSearchNode``: Semantic search
- ``SemIndexNode``: Semantic indexing
- ``SemClusterByNode``: Semantic clustering
- ``SemDedupNode``: Semantic deduplication
- ``SemPartitionByNode``: Semantic partitioning

Pandas Operation Nodes
^^^^^^^^^^^^^^^^^^^^^^

Pandas operations are represented by:

- ``PandasFilterNode``: Boolean filtering
- ``PandasOpNode``: Generic pandas method calls
- ``PandasAssignNode``: Column assignment

Function Application Nodes
^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``ApplyFnNode`` represents function calls that take LazyFrame results:

.. code-block:: python

    from lotus.ast import LazyFrame
    import pandas as pd

    # Concatenate multiple pipelines
    p1 = LazyFrame("data1")
    p2 = LazyFrame("data2")
    combined = LazyFrame.from_fn(pd.concat, [p1, p2])
    result = combined.execute({"data1": df1, "data2": df2})

Execution Engine
----------------

LazyFrameRun
~~~~~~~~~~~~

The ``LazyFrameRun`` class manages the execution of a pipeline with automatic caching:

.. code-block:: python

    from lotus.ast import LazyFrameRun

    run = LazyFrameRun(pipeline, inputs)
    result = run.execute()

Features:

- **Content-Addressable Caching**: Results are cached based on node configuration and input state
- **Shared Cache**: Sub-pipelines (from joins, assigns) share the same cache
- **Cache Statistics**: Track cache hits and misses

Caching
~~~~~~~

The caching system uses content-addressable storage where cache keys are computed from:

1. Node configuration hash (all parameters)
2. Input state hash (current DataFrame state)

This ensures that identical operations on identical data are cached automatically.

Cache keys are computed using:

- ``hash_node()``: Computes a hash from node configuration
- ``hash_dataframe()``: Computes a stable hash for a DataFrame
- ``compute_cache_key()``: Combines node and input hashes

Optimization
------------

The AST module includes an optimization framework for improving pipeline performance.

BaseOptimizer
~~~~~~~~~~~~~

All optimizers inherit from ``BaseOptimizer``:

.. code-block:: python

    from lotus.ast.optimizer import BaseOptimizer
    from lotus.ast.nodes import BaseNode

    class MyOptimizer(BaseOptimizer):
        def optimize_nodes(self, nodes: list[BaseNode]) -> list[BaseNode]:
            # Transform nodes for better performance
            return optimized_nodes

        def get_name(self) -> str:
            return "MyOptimizer"

PredicatePushdownOptimizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``PredicatePushdownOptimizer`` moves pandas filters before semantic filters where safe:

.. code-block:: python

    from lotus.ast import LazyFrame
    from lotus.ast.optimizer import PredicatePushdownOptimizer

    pipeline = (
        LazyFrame("data")
        .sem_filter("text is relevant")
        .filter(lambda df: df["score"] > 0.5)  # This will be moved before sem_filter
    )

    # Apply optimization
    optimized = pipeline.optimize([PredicatePushdownOptimizer()])

    # Or optimize in place
    pipeline.optimize([PredicatePushdownOptimizer()], inplace=True)

This optimization reduces the number of rows processed by expensive semantic operations.

Advanced Features
-----------------

Pipeline Composition
~~~~~~~~~~~~~~~~~~~~

LazyFrames can reference other LazyFrames, enabling complex compositions:

.. code-block:: python

    # Create a base pipeline
    base = LazyFrame("data").sem_filter("is relevant")

    # Create a derived pipeline
    derived = base.sem_map("extract topic")

    # Both pipelines share the same source and can be executed together
    result = derived.execute(input_df)

Joins with Pipelines
~~~~~~~~~~~~~~~~~~~~

Semantic joins can use other pipelines as the right side:

.. code-block:: python

    left = LazyFrame("left_data")
    right = LazyFrame("right_data").sem_filter("is valid")

    # Join with a pipeline
    joined = left.sem_join(
        right,
        "left {description} matches right {title}"
    )

    result = joined.execute({
        "left_data": left_df,
        "right_data": right_df
    })

LazyFrame References in Assignments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use LazyFrames in column assignments:

.. code-block:: python

    base = LazyFrame("data")
    processed = base.sem_map("process {text}")

    # Assign using a LazyFrame
    result = base.assign(processed_text=processed)

    # The processed pipeline will be executed lazily during assignment

Nested Structures
~~~~~~~~~~~~~~~~~

LazyFrames can be nested in lists, tuples, and dictionaries:

.. code-block:: python

    p1 = LazyFrame("data1")
    p2 = LazyFrame("data2")
    p3 = LazyFrame("data3")

    # Concatenate multiple pipelines
    combined = LazyFrame.from_fn(
        pd.concat,
        [p1, p2, p3],
        axis=0,
        ignore_index=True
    )

    result = combined.execute({
        "data1": df1,
        "data2": df2,
        "data3": df3
    })

Self-Referencing Pipelines
~~~~~~~~~~~~~~~~~~~~~~~~~~

The execution engine optimizes self-referencing pipelines by detecting common prefixes:

.. code-block:: python

    base = LazyFrame("data")
    filtered = base.sem_filter("is relevant")
    mapped = filtered.sem_map("extract topic")

    # If mapped references filtered, only the suffix is executed
    result = base.assign(processed=mapped)

Best Practices
--------------

1. **Use Source Keys**: Always provide meaningful source keys for multi-input pipelines
2. **Leverage Caching**: The automatic caching system works best when pipelines are reused
3. **Apply Optimizations**: Use optimizers to improve performance, especially for long pipelines
4. **Inspect Pipelines**: Use ``show()`` to understand pipeline structure before execution
5. **Compose Pipelines**: Build reusable pipeline components that can be combined

Examples
--------

Basic Pipeline
~~~~~~~~~~~~~~

.. code-block:: python

    from lotus.ast import LazyFrame
    import pandas as pd

    # Create data
    df = pd.DataFrame({
        "text": ["I love this product", "This is terrible", "Amazing service"]
    })

    # Build pipeline
    pipeline = (
        LazyFrame("reviews")
        .sem_filter("{text} contains positive sentiment")
        .sem_map("extract the main emotion from {text}")
    )

    # Execute
    result = pipeline.execute({"reviews": df})
    print(result)

Pipeline with Join
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from lotus.ast import LazyFrame
    import pandas as pd

    # Create datasets
    products_df = pd.DataFrame({
        "product_id": [1, 2, 3],
        "description": ["Laptop", "Phone", "Tablet"]
    })

    reviews_df = pd.DataFrame({
        "product_id": [1, 2, 3],
        "review": ["Great laptop", "Good phone", "Nice tablet"]
    })

    # Build pipelines
    products = LazyFrame("products")
    reviews = LazyFrame("reviews")

    # Join pipelines
    joined = products.sem_join(
        reviews,
        "{description} matches {review}",
        how="inner"
    )

    # Execute
    result = joined.execute({
        "products": products_df,
        "reviews": reviews_df
    })
    print(result)

Optimized Pipeline
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from lotus.ast import LazyFrame
    from lotus.ast.optimizer import PredicatePushdownOptimizer
    import pandas as pd

    df = pd.DataFrame({
        "text": ["Positive review", "Negative review", "Neutral review"],
        "score": [0.9, 0.2, 0.5]
    })

    # Build pipeline (filter will be moved before sem_filter by optimizer)
    pipeline = (
        LazyFrame("data")
        .sem_filter("{text} contains positive sentiment")
        .filter(lambda df: df["score"] > 0.5)
    )

    # Optimize
    optimized = pipeline.optimize([PredicatePushdownOptimizer()])

    # Execute
    result = optimized.execute({"data": df})
    print(result)

API Reference
-------------

.. automodule:: lotus.ast
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: lotus.ast.nodes
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: lotus.ast.pipeline
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: lotus.ast.run
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: lotus.ast.cache
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: lotus.ast.optimizer
   :members:
   :undoc-members:
   :show-inheritance:
