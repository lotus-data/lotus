Reasoning Models
=================

Overview
---------
DeepSeek-R1, a lightweight yet powerful reasoning model optimized for CoT-style prompting. 
When using DeepSeek-R1, we recommend setting the temperature between 0.5 and 0.7 (with 0.6 being ideal). 
This range strikes a balance between deterministic reasoning and fluent generation. Lower temperatures (e.g., 0.2) may cause incomplete or overly terse reasoning, 
while higher values (e.g., 0.9+) may lead to hallucination or incoherence.

Filter Example
---------------

.. code-block:: python

    import pandas as pd

    import lotus
    from lotus.models import LM

    lm = LM(model="ollama/deepseek-r1:7b", temperature=0.5)

    lotus.settings.configure(lm=lm)

    data = {
        "Reviews": [
            "I absolutely love this product. It exceeded all my expectations.",
            "Terrible experience. The product broke within a week.",
            "The quality is average, nothing special.",
            "Fantastic service and high quality!",
            "I would not recommend this to anyone.",
        ]
    }
    df = pd.DataFrame(data)
    user_instruction = "{Reviews} are positive reviews"
    df = df.sem_filter(user_instruction, return_explanations=True, return_all=True)

    print(df)

Map Example
------------

.. code-block:: python

    import pandas as pd

    import lotus
    from lotus.models import LM
    from lotus.types import ReasoningStrategy

    lm = LM(model="ollama/deepseek-r1:7b", temperature=0.5)

    lotus.settings.configure(lm=lm)
    data = {
        "Course Name": [
            "Probability and Random Processes",
            "Optimization Methods in Engineering",
            "Digital Design and Integrated Circuits",
            "Computer Security",
        ]
    }
    df = pd.DataFrame(data)
    user_instruction = "What is a similar course to {Course Name}. Just give the course name."
    df = df.sem_map(user_instruction, return_explanations=True, strategy=ReasoningStrategy.ZS_COT)
    print(df)
