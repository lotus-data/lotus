Retrieval Models
==================

Overview
-----------
Any model from the SentenceTransformers can be used with the SentenceTransformerssRM class, by passing
the model name to the model parameter. Additionally, LiteLLM can be used with any model supported by
LiteLLM

Example
----------
Using just the SentenceTransformersRM class

.. code-block:: python

    import pandas as pd

    import lotus
    from lotus.models import SentenceTransformersRM

    rm = SentenceTransformersRM(model="intfloat/e5-base-v2")

    lotus.settings.configure(rm=rm)


Using SentenceTransformersRM and gpt-40-mini

.. code-block:: python
    
    import pandas as pd

    import lotus
    from lotus.models import LM, LiteLLMRM

    lm = LM(model="gpt-4o-mini")
    rm = LiteLLMRM(model="text-embedding-3-small")

    lotus.settings.configure(lm=lm, rm=rm)