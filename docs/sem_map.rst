sem_map
=================

Overview
----------
This operator performs a semantic mapping over input data using natural language instructions. It applies a user-defined instruction to each row of data, transforming the content based on the specified criteria. The operator supports both DataFrame operations and direct function calls on multimodal data.

Motivation
-----------
The sem_map operator is useful for performing row-wise transformations over data using natural language instructions. It enables users to apply complex mappings, transformations, or analyses without writing custom code, making it ideal for tasks like content summarization, sentiment analysis, format conversion, and data enrichment.

Basic Example
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
- **user_instruction** (str): The natural language instruction that guides the mapping process. Should describe how to transform each row. Column names can be referenced using curly braces, e.g., "{column_name}".

Optional Parameters
---------------------
- **system_prompt** (str | None): Custom system prompt to use. Defaults to None.
- **postprocessor** (Callable): Function to post-process model outputs. Should take (outputs, model, use_cot) and return SemanticMapPostprocessOutput. Defaults to map_postprocess.
- **return_explanations** (bool): Whether to include explanations in the output DataFrame. Useful for debugging and understanding model reasoning. Defaults to False.
- **return_raw_outputs** (bool): Whether to include raw model outputs in the output DataFrame. Useful for debugging. Defaults to False.
- **suffix** (str): The suffix for the output column names. Defaults to "_map".
- **examples** (pd.DataFrame | None): Example DataFrame for few-shot learning. Should have the same column structure as the input DataFrame plus an "Answer" column. Defaults to None.
- **strategy** (ReasoningStrategy | None): The reasoning strategy to use. Can be None, COT (Chain-of-Thought), or ZS_COT (Zero-Shot Chain-of-Thought). Defaults to None.
- **safe_mode** (bool): Whether to enable safe mode with cost estimation before execution. Defaults to False.
- **progress_bar_desc** (str): Description for the progress bar. Defaults to "Mapping".
- **model_kwargs**: Additional keyword arguments to pass to the language model.


Return Types and Output Structure
----------------------------------

The sem_map operator returns a DataFrame with the following columns:

- **Original columns**: All original DataFrame columns are preserved
- **{suffix}**: The main output column (default suffix is "_map")
- **explanation{suffix}**: Explanations column (when return_explanations=True)
- **raw_output{suffix}**: Raw model outputs (when return_raw_outputs=True)
