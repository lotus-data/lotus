Tracking LM Usage
=======

Print and Reseet LM Usage Stats
---------
To track usage of the LLM you've configured, you can simply access the built-in ``print_total_usage``

.. code-block:: python

lotus.settings.lm.print_total_usage()


You can also reset the LLM usage stats as follows:

.. code-block:: python

lotus.settings.lm.reset_stats()



Setting Usage Limits
-----------
As a safety measure, the LM class supports setting usage limits to control costs and token consumption. You can set limits on:

- Prompt tokens
- Completion tokens
- Total tokens
- Total cost

When any limit is exceeded, a ``LotusUsageLimitException`` will be raised.

Example setting usage limits:

.. code-block:: python

    from lotus.models import LM
    from lotus.types import UsageLimit, LotusUsageLimitException

    # Set limits
    usage_limit = UsageLimit(
        prompt_tokens_limit=4000,
        completion_tokens_limit=1000,
        total_tokens_limit=3000,
        total_cost_limit=1.00
    )
    lm = LM(model="gpt-4o", physical_usage_limit=usage_limit)

    try:
        course_df = pd.read_csv("course_df.csv")
        course_df = course_df.sem_filter("What {Course Name} requires a lot of math")
    except LotusUsageLimitException as e:
        print(f"Usage limit exceeded: {e}")
        # Handle the exception as needed