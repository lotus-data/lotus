Setting Configurations
=======================

Overview
---------
The Settings module let's you manage application-wide configurations. 
In most examples seen, we have used the settings to configured our LM.

Using the Settings module
--------------------------
.. code-block:: python
    
    from lotus
    from lotus.models import LM

    lm = LM(model="gpt-4o-mini")
    lotus.settings.configure(lm=lm)

Configurable Parameters
--------------------------

1. enable_cache: 
    * Description: Enables or Disables caching mechanisms
    * Default: False
    * Parameters: 
        - cache_type: Type of caching (SQLITE or In_MEMORY)
        - max_size: maximum size of cache
        - cache_dir: Directory for where DB file is stored. Default: "~/.lotus/cache"
    * Note: It is recommended to enable caching
.. code-block:: python

    import pandas as pd

    import lotus
    from lotus.models import LM
    from lotus.cache import CacheFactory, CacheConfig, CacheType
    
    cache_config = CacheConfig(cache_type=CacheType.SQLITE, max_size=1000)
    cache = CacheFactory.create_cache(cache_config)

    lm = LM(model='gpt-4o-mini', cache=cache)
    lotus.settings.configure(lm=lm, enable_cache=True)

2. setting RM:
    * Description: Configures the retrieval model
    * Default: None
.. code-block:: python

    rm = SentenceTransformersRM(model="intfloat/e5-base-v2")
    lotus.settings.configure(rm=rm)

3. setting helper_lm:
    * Descriptions: Configures secondary helper LM often set along with primary LM
    * Default: None
.. code-block:: python

    gpt_4o_mini = LM("gpt-4o-mini")
    gpt_4o = LM("gpt-4o")

    lotus.settings.configure(lm=gpt_4o, helper_lm=gpt_4o_mini)

