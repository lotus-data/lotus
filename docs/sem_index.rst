sem_index
=================

Overview
---------
The sem_index operator in LOTUS creates a semantic index over the specified column in the dataset.
This index enables efficient retrieval and ranking of records based on semantic similarity. 
The index will be generated with the configured retreival model stored locally in the specified directory.


Example
----------
.. code-block:: python

    import pandas as pd

    import lotus
    from lotus.models import LM, CrossEncoderReranker, SentenceTransformersRM

    lm = LM(model="gpt-4o-mini")
    rm = SentenceTransformersRM(model="intfloat/e5-base-v2")
    reranker = CrossEncoderReranker(model="mixedbread-ai/mxbai-rerank-large-v1")

    lotus.settings.configure(lm=lm, rm=rm, reranker=reranker)
    data = {
        "Course Name": [
            "Probability and Random Processes",
            "Optimization Methods in Engineering",
            "Digital Design and Integrated Circuits",
            "Computer Security",
            "Introduction to Computer Science",
            "Introduction to Data Science",
            "Introduction to Machine Learning",
            "Introduction to Artificial Intelligence",
            "Introduction to Robotics",
            "Introduction to Computer Vision",
            "Introduction to Natural Language Processing",
            "Introduction to Reinforcement Learning",
            "Introduction to Deep Learning",
            "Introduction to Computer Networks",
        ]
    }
    df = pd.DataFrame(data)

    df = df.sem_index("Course Name", "index_dir")
    print(df)

Output:

+----+---------------------------------------------+
|    |                Course Name                  |
+----+---------------------------------------------+
|  0 | Probability and Random Processes            |
+----+---------------------------------------------+
|  1 | Optimization Methods in Engineering         |
+----+---------------------------------------------+
|  2 | Digital Design and Integrated Circuits      |
+----+---------------------------------------------+
|  3 | Computer Security                           |
+----+---------------------------------------------+
|  4 | Introduction to Computer Science            |
+----+---------------------------------------------+
|  5 | Introduction to Data Science                |
+----+---------------------------------------------+
|  6 | Introduction to Machine Learning            |
+----+---------------------------------------------+
|  7 | Introduction to Artificial Intelligence     |
+----+---------------------------------------------+
|  8 | Introduction to Robotics                    |
+----+---------------------------------------------+
|  9 | Introduction to Computer Vision             |
+----+---------------------------------------------+
| 10 | Introduction to Natural Language Processing |
+----+---------------------------------------------+
| 11 | Introduction to Reinforcement Learning      |
+----+---------------------------------------------+
| 12 | Introduction to Deep Learning               |
+----+---------------------------------------------+
| 13 | Introduction to Computer Networks           |
+----+---------------------------------------------+


Required Parameters
--------------------
- **col_name** : The column name to index.
- **index_dir** : The directory to save the index.
