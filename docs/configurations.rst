Setting Configurations
=======================

Overview
---------
The Settings module lets you manage application-wide configurations.
Most examples use settings to configure the active LM.

Using the Settings module
--------------------------
.. code-block:: python
    
    import lotus
    from lotus.models import LM

    lm = LM(model="gpt-4o-mini")
    lotus.settings.configure(lm=lm)

Configurable Parameters
--------------------------

``enable_cache``
    Enables or disables caching mechanisms. Default: ``False``.
    Cache parameters include ``cache_type``, ``max_size``, and ``cache_dir``.
    It is recommended to enable caching.

.. code-block:: python

    import pandas as pd

    import lotus
    from lotus.models import LM
    from lotus.cache import CacheFactory, CacheConfig, CacheType
    
    cache_config = CacheConfig(cache_type=CacheType.SQLITE, max_size=1000)
    cache = CacheFactory.create_cache(cache_config)

    lm = LM(model='gpt-4o-mini', cache=cache)
    lotus.settings.configure(lm=lm, enable_cache=True)

``rm``
    Configures the retrieval model. Default: ``None``.

.. code-block:: python

    rm = SentenceTransformersRM(model="intfloat/e5-base-v2")
    lotus.settings.configure(rm=rm)

``helper_lm``
    Configures a secondary helper LM, often used with cascades. Default:
    ``None``.

.. code-block:: python

    gpt_4o_mini = LM("gpt-4o-mini")
    gpt_4o = LM("gpt-4o")

    lotus.settings.configure(lm=gpt_4o, helper_lm=gpt_4o_mini)

Scoped Settings with ``context()``
------------------------------------

``lotus.settings.context(**kwargs)`` is a context manager that temporarily overrides
settings for the duration of a ``with`` block. The previous values are always restored
on exit — even if an exception is raised.

This is useful for:

* Switching to a cheaper model for one step in a pipeline without affecting the rest
* Running an evaluation judge with a fresh model and ``enable_cache=False``
* Isolating settings in tests so one test cannot pollute another
* Running concurrent threads or asyncio tasks with independent settings

Basic usage
~~~~~~~~~~~

.. code-block:: python

    import lotus
    from lotus.models import LM

    lm = LM(model="gpt-4o")
    cheap_lm = LM(model="gpt-4o-mini")
    lotus.settings.configure(lm=lm)

    # Use the cheap model only for this step; gpt-4o is restored afterward
    with lotus.settings.context(lm=cheap_lm, enable_cache=False):
        df = df.sem_filter("Is {Review} positive?")

    # Back to gpt-4o here
    df = df.sem_map("Summarise {Review} in one sentence.")

Nested contexts
~~~~~~~~~~~~~~~

Contexts can be nested. Each level saves and restores independently.

.. code-block:: python

    with lotus.settings.context(lm=cheap_lm):
        # inner context adds another override on top
        with lotus.settings.context(enable_cache=True):
            df = df.sem_map(...)   # cheap_lm + enable_cache=True
        df = df.sem_filter(...)    # cheap_lm only, enable_cache restored

Concurrent threads
~~~~~~~~~~~~~~~~~~

Because ``context()`` uses ``contextvars.ContextVar`` internally, each thread sees
only its own overrides. Threads cannot overwrite each other's settings even though
they share the same ``lotus.settings`` object.

.. code-block:: python

    import threading
    import lotus
    from lotus.models import LM

    lotus.settings.configure(lm=LM(model="gpt-4o-mini"))

    def analyse(df, model, results, key):
        with lotus.settings.context(lm=model):
            results[key] = df.sem_map("Summarise {Text}.")

    results = {}
    t1 = threading.Thread(target=analyse, args=(df1, LM("gpt-4o-mini"), results, "fast"))
    t2 = threading.Thread(target=analyse, args=(df2, LM("gpt-4o"), results, "quality"))
    t1.start(); t2.start()
    t1.join(); t2.join()

Concurrent asyncio tasks
~~~~~~~~~~~~~~~~~~~~~~~~~

``asyncio`` tasks created with ``asyncio.create_task()`` or ``asyncio.gather()`` each
receive a copy of the caller's context, so ``ContextVar`` mutations inside one task
are invisible to others.

.. code-block:: python

    import asyncio
    import lotus
    from lotus.models import LM

    lotus.settings.configure(lm=LM(model="gpt-4o-mini"))

    async def run(df, model):
        with lotus.settings.context(lm=model):
            await asyncio.sleep(0)   # yield; other tasks see their own model
            return df.sem_map("Classify {Text}.")

    async def main():
        results = await asyncio.gather(
            run(df1, LM("gpt-4o-mini")),
            run(df2, LM("gpt-4o")),
        )

    asyncio.run(main())

.. note::

    ``configure()`` mutates the global settings object and is **not** thread-safe.
    Use ``context()`` whenever settings need to differ across concurrent execution paths.
