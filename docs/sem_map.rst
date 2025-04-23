sem_map
=================

Overview
----------
This operator performs a semantic projection over an input column. The langex parameter specifies this projection in natural language. It can generate a single output or multiple sample outputs for each input.

Motivation
-----------
The sem_map operator is useful for performing row-wise operations over the data. The multi-sampling capability allows for generating diverse outputs for the same input, which can be useful for creative tasks, exploring multiple possibilities, or for understanding the variability in model outputs.

Example
----------
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
    user_instruction = "What is a similar course to {Course Name}. Be concise."
    df = df.sem_map(user_instruction)
    print(df)
    
    # Example with multiple samples and temperature
    df = df.sem_map(user_instruction, nsample=3, temp=0.7)
    print(df)

Output:

+---+----------------------------------------+----------------------------------------------------------------+
|   | Course Name                            | _map                                                           |
+===+========================================+================================================================+
| 0 | Probability and Random Processes       | A similar course to "Probability and Random Processes"...      |
+---+----------------------------------------+----------------------------------------------------------------+
| 1 | Optimization Methods in Engineering    | A similar course to "Optimization Methods in Engineering"...   |
+---+----------------------------------------+----------------------------------------------------------------+
| 2 | Digital Design and Integrated Circuits | A similar course to "Digital Design and Integrated Circuits"...|
+---+----------------------------------------+----------------------------------------------------------------+
| 3 | Computer Security                      | A similar course to "Computer Security" is "Cybersecurity"...  |
+---+----------------------------------------+----------------------------------------------------------------+

Required Parameters
---------------------
- **user_instruction** : The user instruction for map.
- **postprocessor** : The postprocessor for the model outputs. Defaults to map_postprocess.

Optional Parameters
---------------------
- **return_explanations** : Whether to return explanations. Defaults to False.
- **return_raw_outputs** : Whether to return raw outputs. Defaults to False.
- **suffix** : The suffix for the new columns. Defaults to "_map".
- **examples** : The examples dataframe. Defaults to None.
- **strategy** : The reasoning strategy. Defaults to None.
- **nsample** : Number of samples to generate per input. Defaults to 1.
- **temp** : Temperature for sampling. Higher values (e.g., 0.7, 1.0) increase randomness in the output, while lower values (e.g., 0.0, 0.1) make the output more deterministic. Defaults to None (using the model's default temperature).

Multiple Sample Output Structure
--------------------------------
When using ``nsample > 1``, the output dataframe will contain multiple columns, one for each sample:

- With ``nsample=3`` and ``suffix="_map"``, the output columns will be "_map1", "_map2", and "_map3"
- If ``return_explanations=True``, the explanation columns will be "explanation_map1", "explanation_map2", and "explanation_map3"
- If ``return_raw_outputs=True``, the raw output columns will be "raw_output_map1", "raw_output_map2", and "raw_output_map3"

Examples
--------

Basic example with multiple samples:

.. code-block:: python

    # Generate 5 different responses per row, with increased randomness
    df = df.sem_map("Summarize {article} in one sentence", nsample=5, temp=0.8)
    
    # Access the different samples
    print(df["_map1"])  # First sample for each row
    print(df["_map2"])  # Second sample for each row
