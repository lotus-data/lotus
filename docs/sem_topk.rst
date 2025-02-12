sem_topk
================

Overview
---------
LOTUS supports a semantic top-k, which takes the langex ranking criteria. Programmers can optionally 
specify a group-by parameter to indicate a subset of columns to group over during ranking. 
The groupings are defined using standard equality matches over the group-by columns

Motivation
-----------
This operator is useful for re-ordering records based on complex, arbitrary natural language comparators.

Example
--------
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

    for method in ["quick", "heap", "naive"]:
        sorted_df, stats = df.sem_topk(
            "Which {Course Name} requires the least math?",
            K=2,
            method=method,
            return_stats=True,
        )
        print(sorted_df)
        print(stats)


Output:

+---+----------------------------------------+
|   |           Course Name                  |
+---+----------------------------------------+
| 0 | Computer Security                      |
+---+----------------------------------------+
| 1 | Digital Design and Integrated Circuits |
+---+----------------------------------------+

Required Parameters
--------------------
- **user_instruction** : The user instruction for sorting.
- **K**: The number of rows to return.

Optional Paramaters
---------------------
- **method** : The method to use for sorting. Options are "quick", "heap", "naive", "quick-sem".
- **group_by** : The columns to group by before sorting. Each group will be sorted separately.
- **cascade_threshold**: The confidence threshold for cascading to a larger model.
- **return_stats** : Whether to return stats.