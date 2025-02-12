Core Concepts
==================

LOTUS' implements the semantic operator programming model. Semantic operators are declarative transformations over one or more
datasets, parameterized by a natural langauge expression (*langex*) that can be implemnted by a variety of AI-based algorithms.
Semantic operators seamlessly extend the relational model, operating over datasets that may contain traditional structured data
as well as unstructured fields, such as free-form text or images. Because semantic operators are composable, modular and declarative, they allow you to write 
AI-based piplines with intuitive, high-level logic, leaving the rest of the work to the query engine! Each operator can be implmented and 
optimized in multiple ways, opening a rich space for execution plans, similar to relational operators. Here is a quick example of semantic operators in action:

.. code-block:: python

    langex = "The {abstract} suggests that LLMs efficeintly utilize long context"
    filtered_df = papers_df.sem_filter(langex)


With LOTUS, applications can be built by chaining togethor different semantic operators. Much like relational operators, semantic operators represent transformations over the dataset, and can be implemented and optimized under the hood. Each semantic operator is parameterized by a natural language expression. 
Here are some key semantic operators:

+--------------+----------------------------------------------------------+
| Operator     | Description                                              |
+==============+==========================================================+
| sem_map      |  Map each record using a natural language projection     |                
+--------------+----------------------------------------------------------+
| sem_extract  | Extract one or more attributes from each row             |
+--------------+----------------------------------------------------------+
| sem_filter   | Keep records that match the natural language predicate   |                  
+--------------+----------------------------------------------------------+
| sem_agg      | Aggregate across all records (e.g. for summarization)    |
+--------------+----------------------------------------------------------+
| sem_topk     | Order records by the natural langauge ranking criteria   |
+--------------+----------------------------------------------------------+
| sem_join     | Join two datasets based on a natural language predicate  |
+--------------+----------------------------------------------------------+
| sem_sim_join | Join two DataFrames based on semantic similarity         |
+--------------+----------------------------------------------------------+
| sem_search   | Perform semantic search the over a text column           |
+--------------+----------------------------------------------------------+

