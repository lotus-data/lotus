Setting Configurations
=======================

Overview
---------
The Settings module is a central configuration system for managing application-wide settings. 
It ensures consistent and thread-safe access to configurations, allowing settings to be dynamically 
adjusted and temporarily overridden within specific contexts. In most examples seen, we have 
used the settings to configured our LM.

Using the Settings module
--------------------------
.. code-block:: python
    
    from lotus
    from lotus.models import LM

    lm = LM(model="gpt-4o-mini")
    lotus.settings.configure(lm=lm)

Configurable Parameters
--------------------------

1. enable_message_cache: 
    * Description: Enables or Disables cahcing mechanisms
    * Default: False
.. code-block:: python

    lotus.settings.configure(enable_message_cache=True)

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

