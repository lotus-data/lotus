LLM
=======

Overview
---------
The LM class is built on top of the LiteLLM library, and supports any model that is supported by LiteLLM.
Example models include but not limited to: OpenAI, Ollama, vLLM

Example
---------
To run a model, you can use the LM class. We use the LiteLLM library to interface with the model. This allows
you to use any model provider that is supported by LiteLLM.

Creating a LM object for gpt-4o

.. code-block:: python

    from lotus.models import LM
    lm = LM(model="gpt-4o")

Creating a LM object to use llama3.2 on Ollama

.. code-block:: python

    from lotus.models import LM
    lm = LM(model="ollama/llama3.2")

Creating a LM object to use Meta-Llama-3-8B-Instruct on vLLM

.. code-block:: python

    from lotus.models import LM
    lm = LM(model='hosted_vllm/meta-llama/Meta-Llama-3-8B-Instruct',
        api_base='http://localhost:8000/v1',
        max_ctx_len=8000,
        max_tokens=1000)
