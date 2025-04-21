Vector Stores
=====================

Lotus supports multiple vector store backends for efficient semantic indexing and search. This document describes how to use and configure the available vector stores, including Qdrant, Faiss, and Weaviate.

Supported Vector Stores
----------------------
- QdrantVS
- FaissVS
- WeaviateVS

QdrantVS
--------

**Installation**
^^^^^^^^^^^^^^^^
Install the Qdrant client and Lotus with Qdrant support:

.. code-block:: bash

    pip install qdrant-client lotus[qdrant]

**Running Qdrant**
^^^^^^^^^^^^^^^^^^
You can run Qdrant locally using Docker:

.. code-block:: bash

    docker run -p 6333:6333 -p 6334:6334 \
        -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
        qdrant/qdrant

**Example Usage**
^^^^^^^^^^^^^^^^^

.. code-block:: python

    import pandas as pd
    from qdrant_client import QdrantClient
    import lotus
    from lotus.models import LiteLLMRM  # or SentenceTransformersRM
    from lotus.vector_store import QdrantVS

    # Start Qdrant server before running this code
    client = QdrantClient(url="http://localhost:6333")
    rm = LiteLLMRM(model="text-embedding-3-small")
    vs = QdrantVS(client)
    lotus.settings.configure(rm=rm, vs=vs)

    data = {"Course Name": ["Machine Learning 101", "Introduction to Cooking"]}
    df = pd.DataFrame(data)
    df = df.sem_index("Course Name", "my_qdrant_index")
    result = df.sem_search("Course Name", "Find the course about machine learning", K=1)
    print(result)

FaissVS
-------

**Installation**
^^^^^^^^^^^^^^^^

.. code-block:: bash

    pip install faiss-cpu lotus

**Example Usage**
^^^^^^^^^^^^^^^^^

.. code-block:: python

    import pandas as pd
    import lotus
    from lotus.models import LiteLLMRM
    from lotus.vector_store import FaissVS

    rm = LiteLLMRM(model="text-embedding-3-small")
    vs = FaissVS()
    lotus.settings.configure(rm=rm, vs=vs)

    data = {"Course Name": ["Machine Learning 101", "Introduction to Cooking"]}
    df = pd.DataFrame(data)
    df = df.sem_index("Course Name", "my_faiss_index")
    result = df.sem_search("Course Name", "Find the course about machine learning", K=1)
    print(result)

WeaviateVS
----------

**Installation**
^^^^^^^^^^^^^^^^

.. code-block:: bash

    pip install weaviate-client lotus[weaviate]

**Running Weaviate**
^^^^^^^^^^^^^^^^^^^^
You can run Weaviate locally using Docker:

.. code-block:: bash

    docker run -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:1.29.1

**Example Usage**
^^^^^^^^^^^^^^^^^

.. code-block:: python

    import pandas as pd
    import weaviate
    import lotus
    from lotus.models import LiteLLMRM
    from lotus.vector_store import WeaviateVS

    client = weaviate.Client("http://localhost:8080")
    rm = LiteLLMRM(model="text-embedding-3-small")
    vs = WeaviateVS(client)
    lotus.settings.configure(rm=rm, vs=vs)

    data = {"Course Name": ["Machine Learning 101", "Introduction to Cooking"]}
    df = pd.DataFrame(data)
    df = df.sem_index("Course Name", "my_weaviate_index")
    result = df.sem_search("Course Name", "Find the course about machine learning", K=1)
    print(result)
