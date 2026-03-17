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

.. code-block:: console

    # Install uv if you haven't already
    $ curl -LsSf https://astral.sh/uv/install.sh | sh

    # Create a new project or navigate to your existing project
    $ uv add lotus-ai

For the latest features:

.. code-block:: console

    $ uv add git+https://github.com/lotus-data/lotus.git@main

Install with pip
----------------

For the latest stable release:

.. code-block:: console

    $ conda create -n lotus python=3.10 -y
    $ conda activate lotus
    $ pip install lotus-ai

For the latest features:

.. code-block:: console

    $ conda create -n lotus python=3.10 -y
    $ conda activate lotus
    $ pip install git+https://github.com/lotus-data/lotus.git@main

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
