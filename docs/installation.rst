Installation
============

LOTUS can be installed as a Python library through pip or uv.

Requirements
------------

* OS: MacOS, Linux
* Python: 3.10

Install with uv (Recommended)
------------------------------

For the latest stable release:

.. code-block:: bash

    # Install uv if you haven't already
    $ curl -LsSf https://astral.sh/uv/install.sh | sh

    # Create a new project or navigate to your existing project
    $ uv add lotus-ai

For the latest features:

.. code-block:: bash

    $ uv add git+https://github.com/lotus-data/lotus.git@main

Install with pip
----------------

For the latest stable release:

.. code-block:: bash

    $ conda create -n lotus python=3.10 -y
    $ conda activate lotus
    $ pip install lotus-ai

For the latest features:

.. code-block:: bash

    $ conda create -n lotus python=3.10 -y
    $ conda activate lotus
    $ pip install git+https://github.com/lotus-data/lotus.git@main

Optional Subpackages
--------------------

LOTUS supports optional subpackages for extended functionality. Install them using the ``lotus-ai[<subpackage>]`` syntax:

.. code-block:: bash

    $ pip install "lotus-ai[serpapi]"

Or with uv:

.. code-block:: bash

    $ uv add "lotus-ai[serpapi]"

Here's a non-exhaustive list of available subpackages:

* ``serpapi`` — Web search via Google and Google Scholar using the SerpAPI
* ``arxiv`` — Fetching and searching papers from arXiv
* ``pubmed`` — Searching and extracting biomedical literature via PubMed
* ``file_extractor`` — Extracting text from PDFs, Word documents, PowerPoint files, and more via DirectoryReader
* ``weaviate`` — Integration with the Weaviate vector database
* ``qdrant`` — Integration with the Qdrant vector database
* ``data_connectors`` — Connecting to SQL databases and AWS S3 via SQLAlchemy and boto3
* ``web_search`` — Web search support via You.com and Tavily

Running on Mac
--------------

If you are running on Mac and using pip, please install Faiss via conda:

.. code-block:: console

    # CPU-only version
    $ conda install -c pytorch faiss-cpu=1.8.0

    # GPU(+CPU) version
    $ conda install -c pytorch -c nvidia faiss-gpu=1.8.0

If you're using uv, the faiss-cpu dependency will be handled automatically.

For more details, see `Installing FAISS via Conda <https://github.com/facebookresearch/faiss/blob/main/INSTALL.md#installing-faiss-via-conda>`_.
