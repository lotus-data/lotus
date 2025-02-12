sem_dedup
========================

Overview
---------
Semantic deduplication is a process designed to identify and eliminate semantically 
redundant entries from datasets, focusing on meaning rather than exact textual matches. 
Entity de-duplication can be implemented as a semantic self-join, but we provide an additional utility function.

Motivation
-----------
Unlike traditional deduplication techniques, which rely on exact or near-exact string comparisons, 
semantic deduplication uses language models to compare the underlying meaning of text entries. 
This ensures that even paraphrased or contextually similar items can be identified as duplicates.

Example
--------
.. code-block:: python

    import pandas as pd

    import lotus
    from lotus.models import SentenceTransformersRM

    rm = SentenceTransformersRM(model="intfloat/e5-base-v2")

    lotus.settings.configure(rm=rm)
    data = {
        "Text": [
            "Probability and Random Processes",
            "Optimization Methods in Engineering",
            "Digital Design and Integrated Circuits",
            "Computer Security",
            "I don't know what day it is",
            "I don't know what time it is",
            "Harry potter and the Sorcerer's Stone",
        ]
    }
    df = pd.DataFrame(data)
    df = df.sem_index("Text", "index_dir").sem_dedup("Text", threshold=0.815)
    print(df)

Output:

+---+------------------------------------------+
|   |                   Text                   |
+---+------------------------------------------+
| 0 | Probability and Random Processes         |
+---+------------------------------------------+
| 5 | I don't know what time it is             |
+---+------------------------------------------+
| 6 | Harry Potter and the Sorcerer's Stone    |
+---+------------------------------------------+

Required Parameters
--------------------
- **col_name** : The column name to deduplicate on
- **threshold** : The threshold for similarity score

