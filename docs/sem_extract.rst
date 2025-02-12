sem_extract
==================

Overview
---------
The sem_extract operator generates one or more columns from the input columns. 
Each output columns is specified by a natural language projection.
Optionally, you can also extract direct quotes from the source text to support each output. 


Motivation
-----------
Semantic extractions can be useful for generating structured schemas that provide a simplified view of the data from a column of unstructured documents.
The quoting functionality can also be useful for tasks, such as entity extraction or fact-checking, where finding snippets or verified quotes 
may be preferable to synthesized answers.


Example
--------
.. code-block:: python

    import pandas as pd

    import lotus
    from lotus.models import LM

    lm = LM(model="gpt-4o-mini")
    lotus.settings.configure(lm=lm)

    df = pd.DataFrame(
        {
            "description": [
                "Yoshi is 25 years old",
                "Bowser is 45 years old",
                "Luigi is 15 years old",
            ]
        }
    )
    input_cols = ["description"]

    # A description can be specified for each output column
    output_cols = {
        "masked_col_1": "The name of the person",
        "masked_col_2": "The age of the person",
    }

    # you can optionally set extract_quotes=True to return quotes that support each output
    new_df = df.sem_extract(input_cols, output_cols, extract_quotes=True) 
    print(new_df)

    # A description can also be omitted for each output column
    output_cols = {
        "name": None,
        "age": None,
    }
    new_df = df.sem_extract(input_cols, output_cols)
    print(new_df)

Output:

+---+--------------------------+---------------+---------------+---------------------+---------------------+
|   |       description        | masked_col_1  | masked_col_2  | masked_col_1_quote  | masked_col_2_quote  |
+===+==========================+===============+===============+=====================+=====================+
| 0 | Yoshi is 25 years old    | Yoshi         | 25            | Yoshi               | 25 years old        |
+---+--------------------------+---------------+---------------+---------------------+---------------------+
| 1 | Bowser is 45 years old   | Bowser        | 45            | Bowser              | 45 years old        |
+---+--------------------------+---------------+---------------+---------------------+---------------------+
| 2 | Luigi is 15 years old    | Luigi         | 15            | Luigi               | 15 years old        |
+---+--------------------------+---------------+---------------+---------------------+---------------------+

+---+--------------------------+---------------+---------------+
|   |       description        | masked_col_1  | masked_col_2  |
+===+==========================+===============+===============+
| 0 | Yoshi is 25 years old    | Yoshi         | 25            |
+---+--------------------------+---------------+---------------+
| 1 | Bowser is 45 years old   | Bowser        | 45            |
+---+--------------------------+---------------+---------------+
| 2 | Luigi is 15 years old    | Luigi         | 15            |
+---+--------------------------+---------------+---------------+


Required Parameters
--------------------
- **input_cols** : The columns that a model should extract from.
- **output_cols** : A mapping from desired output column names to optional descriptions.

Optional Parameters
--------------------
- **extract_quotes** : Whether to extract quotes for the output columns. Defaults to False.
- **postprocessor** : The postprocessor for the model outputs. Defaults to extract_postprocess.
- **return_raw_outputs** : Whether to return raw outputs. Defaults to False.