ReRanker Models
==================

Overview
----------
Any CrossEncoder from SentenceTransfomers can be used with the CrossEncoderReranker class, by
passing the model name to the model parameter. The LM class and Retrieval model class can also be passed
to the model parameter

Example
--------
Passing the LM, Retrieval, and ReRanker to model parameters

.. code-block:: python

    import pandas as pd

    import lotus
    from lotus.models import LM, CrossEncoderReranker, SentenceTransformersRM

    lm = LM(model="gpt-4o-mini")
    rm = SentenceTransformersRM(model="intfloat/e5-base-v2")
    reranker = CrossEncoderReranker(model="mixedbread-ai/mxbai-rerank-large-v1")

    lotus.settings.configure(lm=lm, rm=rm, reranker=reranker)