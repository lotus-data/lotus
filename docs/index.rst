.. lotus documentation master file, created by
   sphinx-quickstart on Sun May 12 13:55:36 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: logo_with_text.png
   :width: 300px
   :height: 170px
   :align: center

LOTUS: Optimized Agentic and LLM Bulk Processing
=================================================================================

LOTUS makes agentic and LLM bulk processing fast, easy, and robust. It
introduces semantic operators (e.g., map, reduce, filter primitives) for processing structured and unstructured data corpora at scale with parallel agents and LLM calls.
LOTUS' optimized query engine allows you to write declarative code for complex data processing tasks with higher accuracy and lower cost.

LOTUS supports two classes of semantic operators. **Agentic semantic operators**
(``corpus.agent(ops=[...])``) run tool-using agents over a corpus and are built for
complex or ambiguous tasks that benefit from multiple steps and tool calls. **LLM
semantic operators** (``sem_map``, ``sem_filter``, ``sem_agg``, ``sem_join``, …) invoke
far fewer model calls per record and are ideal for well-defined tasks such as
LLM-as-judge evaluation, document extraction, and unstructured data analysis.

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting Started

   installation
   core_concepts

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Agentic Semantic Operators

   agentic_operators
   corpus
   agentic_map_reduce
   agentic_examples
   agentic_filter

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: LLM Semantic Operators

   sem_map
   sem_extract
   sem_filter
   sem_agg
   sem_topk
   sem_join
   sem_search
   sem_sim_join
   sem_cluster

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Optimizations

   lazyframe
   lazyframe_optimizations
   lazyframe_api

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: LLM Judge Suite

   evaluation
   llm_as_judge
   pairwise_judge
   evaluation_advanced

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Utility Operators

   sem_partition
   sem_index
   sem_dedup
   web_search
   web_extract

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Models

   llm
   retriever_models
   reranker_models
   multimodal_models
   vector_store
   usage

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Advanced Usage

   approximation_cascades
   prompt_strategies
   configurations
   reasoning_models

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Data Loading and DB Connectors

   data_connectors
   DirectoryReader
