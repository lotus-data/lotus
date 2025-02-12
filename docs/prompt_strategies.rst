Prompt Strategies
===================

Overview
----------
In addition to calling the semantic operators, advanced prompt stratigies can be used to potentially
get or improve the desired output. Two Prompt Strategies that can be used are Chain of Thought (CoT) and 
Demonstrations.

Chain of Thought + Demonstrations:
----------------------------------
Chain of Thought reasoning refers to structing prompts in a way that guides the model through a step-by-step process 
to arrive at a final answer. By breaking down complex tasks into intermediate steps, CoT ensures more accurate and 
logical output

Here is a simple example of using chain of thought with the Semantic Filter operator

.. code-block:: python

    import pandas as pd

    import lotus
    from lotus.models import LM

    lm = LM(model="gpt-4o-mini")

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
    user_instruction = "{Course Name} requires a lot of math"

    example_data = {
        "Course Name": ["Machine Learning", "Reaction Mechanisms", "Nordic History"], 
        "Answer": [True, True, False],
        "Reasoning": ["Machine Learning requires a solid understanding of linear alebra and calculus",
                      "Reaction Engineering requires Ordinary Differential Equations to solve reactor design problems",
                      "Nordic History has no math involved"]
    }
    examples = pd.DataFrame(example_data)

    df = df.sem_filter(user_instruction, examples=examples, strategy="cot")
    print(df)

When calling the Semantic Filter operator, we pass in an example DataFrame as well as the CoT strategy, which acts as a guide 
for how the model should reason and respond to the given instructions. For instance, in the examples DataFrame 

* "Machine Learning" has an answer of True, with reasoning that it requires a solid understanding of linear algebra and calculus.
* "Reaction Mechanisms" also has an answer of True, justified by its reliance on ordinary differential equations for solving reactor design problems.
* "Nordic History" has an answer of False, as it does not involve any mathematical concepts.

Using the CoT strategy will provide an output below:

+---+----------------------------------------+-------------------------------------------------------------------+
|   |           Course Name                  |                    explanation_filter                             |
+---+----------------------------------------+-------------------------------------------------------------------+
| 0 | Probability and Random Processes       | Probability and Random Processes is heavily based on...           |
+---+----------------------------------------+-------------------------------------------------------------------+
| 1 | Optimization Methods in Engineering    | Optimization Methods in Engineering typically involves...         |
+---+----------------------------------------+-------------------------------------------------------------------+
| 2 | Digital Design and Integrated Circuits | Digital Design and Integrated Circuits typically covers...        |
+---+-------------------------------------+----------------------------------------------------------------------+