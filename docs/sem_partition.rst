sem_partition_by
====================

Overview
---------
The sem_partition_by utility in LOTUS exposes finer-grained control over how data is processed for sem_agg.
This operator let's you assign a partition number to each row in a DataFrame. During semantic aggregation, LOTUS, will aggregate over each partition separately,
before combining intermediate aggregations across partitions. Additionally, the order in which each partition aggregates is combined will follow the order of the partition numbers in increasing order.
By default, LOTUS implements a hierarchical reduce strategy, assuming that all record belong to the same partition.

Motivation
----------
Since LLMs are sensitive to the ordering of inputs, specifying an aggregation ordering using sem_partition_by can provide fine-grained control to achieve high quality results for tasks like summarization.


Example
----------
.. code-block:: python
    
    import pandas as pd

    import lotus
    from lotus.models import LM, SentenceTransformersRM

    lm = LM(max_tokens=2048)
    rm = SentenceTransformersRM(model="intfloat/e5-base-v2")

    lotus.settings.configure(lm=lm, rm=rm)
    data = {
        "Course Name": [
            "Probability and Random Processes",
            "Optimization Methods in Engineering",
            "Digital Design and Integrated Circuits",
            "Computer Security",
            "Cooking",
            "Food Sciences",
        ]
    }
    df = pd.DataFrame(data)
    df = df.sem_index("Course Name", "course_name_index").sem_partition_by(lotus.utils.cluster("Course Name", 2))
    out = df.sem_agg("Summarize all {Course Name}")._output[0]
    print(out)


Required Parameters
--------------------
- **partition_fn** : The partitioning function, which returns a list[int], indicating the partition-id of each row.


