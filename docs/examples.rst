Examples
==================

Let's walk through some use cases of LOTUS.
First let's configure LOTUS to use GPT-3.5-Turbo for the LLM and E5 as the embedding model.
Then let's define a dataset of courses and their descriptions/workloads.
Next let's use LOTUS to filter for machine learning courses and then summarize how to succeed in them.
This can be achieved by applying a semantic filter followed by a semantic aggregation.

.. code-block:: python

    import pandas as pd

    import lotus
    from lotus.models import SentenceTransformersRM, LM

    # Configure models for LOTUS
    lm = LM(model="gpt-4o-mini")
    rm = SentenceTransformersRM(model="intfloat/e5-base-v2")

    lotus.settings.configure(lm=lm, rm=rm)

    # Dataset containing courses and their descriptions/workloads
    data = [
        (
            "Probability and Random Processes",
            "Focuses on markov chains and convergence of random processes. The workload is pretty high.",
        ),
        (
            "Deep Learning",
            "Fouces on theory and implementation of neural networks. Workload varies by professor but typically isn't terrible.",
        ),
        (
            "Digital Design and Integrated Circuits",
            "Focuses on building RISC-V CPUs in Verilog. Students have said that the workload is VERY high.",
        ),
        (
            "Databases",
            "Focuses on implementation of a RDBMS with NoSQL topics at the end. Most students say the workload is not too high.",
        ),
    ]
    df = pd.DataFrame(data, columns=["Course Name", "Description"])

    # Applies semantic filter followed by semantic aggregation
    ml_df = df.sem_filter("{Description} indicates that the class is relevant for machine learning.")
    tips = ml_df.sem_agg(
        "Given each {Course Name} and its {Description}, give me a study plan to succeed in my classes."
    )._output[0]


If we wanted the challenge of taking courses with a high workload, we can also use the semantic top k operator to get the top 2 courses with the highest workload.

.. code-block:: python

    top_2_hardest = df.sem_topk("What {Description} indicates the highest workload?", K=2)

LOTUS's semantic join operator can be used to join two dataframes based on a predicate.
Suppose we had a second dataframe containing skills we wanted to get better at (SQL and Chip Design in our case).
We can use LOTUS's semantic join to find courses that will help us improve those skills.

.. code-block:: python

    skills_df = pd.DataFrame(
        [("SQL"), ("Chip Design")], columns=["Skill"]
    )
    classes_for_skills = skills_df.sem_join(
        df, "Taking {Course Name} will make me better at {Skill}"
    )

Two other powerful operators are the semantic index and search operators.
The semantic index operator allows us to index a dataframe based on a column, while the semantic search operator allows us to search for relevant rows using the index and a query.
Let's create a semantic index on the course description column and then search for the class that is most relevant for convolutional neural networks.

.. code-block:: python

    # Create a semantic index on the description column and save it to the index_dir directory
    df = df.sem_index("Description", "index_dir")
    top_conv_df = df.sem_search("Description", "Convolutional Neural Network", K=1)

Another useful operator is the semantic map operator. Let's see how it can be used to get some next topics to explore for each class.
Additionally, let's provide some examples to the model that can be used for demonstrations.

.. code-block:: python

    examples_df = pd.DataFrame(
        [("Computer Graphics", "Computer Vision"), ("Real Analysis", "Complex Analysis")],
        columns=["Course Name", "Answer"]
    )
    next_topics = df.sem_map(
        "Given {Course Name}, list a topic that will be good to explore next. \
        Respond with just the topic name and nothing else.", examples=examples_df, suffix="Next Topics"
    )

Now you've seen how to use LOTUS to implement LLM-powered transformations in a couple of steps using semantic operators in LOTUS!