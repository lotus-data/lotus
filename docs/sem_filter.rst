sem_filter
=================

Overview
---------
sem_filter take a langex predicate, and returns data records that pass the predicate. 

Motivation
-----------
Semantic filtering is a complex yet vital operation in modern data processing, requiring accurate and efficient 
evaluation of data rows against nuanced, natural language predicates. Unlike traditional filtering techniques, 
which rely on rigid and often simplistic rules, semantic filters must leverage language models to reason contextually about the data. 


Filter Example
---------------
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
    df = df.sem_filter(user_instruction)
    print(df)

Output:

+---+---------------------------------------------+
|   |                Course Name                  |
+---+---------------------------------------------+
| 0 | Probability and Random Processes            |
+---+---------------------------------------------+
| 1 | Optimization Methods in Engineering         |
+---+---------------------------------------------+
| 2 | Digital Design and Integrated Circuits      |
+---+---------------------------------------------+



Example of Filter with Approximation
-----------------------
.. code-block:: python

    import pandas as pd

    import lotus
    from lotus.models import LM
    from lotus.types import CascadeArgs


    gpt_4o_mini = LM("gpt-4o-mini")
    gpt_4o = LM("gpt-4o")

    lotus.settings.configure(lm=gpt_4o, helper_lm=gpt_4o_mini)
    data = {
        "Course Name": [
            "Probability and Random Processes", "Optimization Methods in Engineering", "Digital Design and Integrated Circuits",
            "Computer Security", "Data Structures and Algorithms", "Machine Learning", "Artificial Intelligence", "Natural Language Processing",
            "Introduction to Robotics", "Control Systems", "Linear Algebra and Differential Equations", "Database Systems", "Cloud Computing",
            "Software Engineering", "Operating Systems", "Discrete Mathematics", "Numerical Methods", "Wireless Communication Systems",
            "Embedded Systems", "Advanced Computer Architecture", "Graph Theory", "Cryptography and Network Security",
            "Big Data Analytics", "Deep Learning", "Organic Chemistry", "Molecular Biology", "Environmental Science",
            "Genetics and Evolution", "Human Physiology", "Introduction to Anthropology", "Cultural Studies", "Political Theory",
            "Macroeconomics", "Microeconomics", "Introduction to Sociology", "Developmental Psychology", "Cognitive Science",
            "Introduction to Philosophy", "Ethics and Moral Philosophy", "History of Western Civilization", "Art History: Renaissance to Modern",
            "World Literature", "Introduction to Journalism", "Public Speaking and Communication", "Creative Writing", "Music Theory",
            "Introduction to Theater", "Film Studies", "Environmental Policy and Law", "Sustainability and Renewable Energy",
            "Urban Planning and Design", "International Relations", "Marketing Principles", "Organizational Behavior",
            "Financial Accounting", "Corporate Finance", "Business Law", "Supply Chain Management", "Operations Research",
            "Entrepreneurship and Innovation", "Introduction to Psychology", "Health Economics", "Biostatistics",
            "Social Work Practice", "Public Health Policy", "Environmental Ethics", "History of Political Thought", "Quantitative Research Methods",
            "Comparative Politics", "Urban Economics", "Behavioral Economics", "Sociology of Education", "Social Psychology",
            "Gender Studies", "Media and Communication Studies", "Advertising and Brand Strategy",
            "Sports Management", "Introduction to Archaeology", "Ecology and Conservation Biology", "Marine Biology",
            "Geology and Earth Science", "Astronomy and Astrophysics", "Introduction to Meteorology",
            "Introduction to Oceanography", "Quantum Physics", "Thermodynamics", "Fluid Mechanics", "Solid State Physics",
            "Classical Mechanics", "Introduction to Civil Engineering", "Material Science and Engineering", "Structural Engineering",
            "Environmental Engineering", "Energy Systems Engineering", "Aerodynamics", "Heat Transfer",
            "Renewable Energy Systems", "Transportation Engineering", "Water Resources Management", "Principles of Accounting",
            "Project Management", "International Business", "Business Analytics",
        ]
    }
    df = pd.DataFrame(data)
    user_instruction = "{Course Name} requires a lot of math"

    cascade_args = CascadeArgs(recall_target=0.9, precision_target=0.9, sampling_percentage=0.5, failure_probability=0.2)

    df, stats = df.sem_filter(user_instruction=user_instruction, cascade_args=cascade_args, return_stats=True)
    print(df)
    print(stats)

Output:

+-----+---------------------------------------------+
|     |                Course Name                  |
+-----+---------------------------------------------+
|   0 | Probability and Random Processes            |
+-----+---------------------------------------------+
|   1 | Optimization Methods in Engineering         |
+-----+---------------------------------------------+
|   2 | Digital Design and Integrated Circuits      |
+-----+---------------------------------------------+
|   5 | Machine Learning                            |
+-----+---------------------------------------------+
|   6 | Artificial Intelligence                     |
+-----+---------------------------------------------+
|   7 | Natural Language Processing                 |
+-----+---------------------------------------------+
|   8 | Introduction to Robotics                    |
+-----+---------------------------------------------+
|   9 | Control Systems                             |
+-----+---------------------------------------------+
|  10 | Linear Algebra and Differential Equations   |
+-----+---------------------------------------------+
|  15 | Discrete Mathematics                        |
+-----+---------------------------------------------+
|  16 | Numerical Methods                           |
+-----+---------------------------------------------+
|  17 | Wireless Communication Systems              |
+-----+---------------------------------------------+
|  19 | Advanced Computer Architecture              |
+-----+---------------------------------------------+
|  20 | Graph Theory                                |
+-----+---------------------------------------------+
|  21 | Cryptography and Network Security           |
+-----+---------------------------------------------+
|  22 | Big Data Analytics                          |
+-----+---------------------------------------------+
|  23 | Deep Learning                               |
+-----+---------------------------------------------+
|  33 | Microeconomics                              |
+-----+---------------------------------------------+
|  55 | Corporate Finance                           |
+-----+---------------------------------------------+
|  58 | Operations Research                         |
+-----+---------------------------------------------+
|  61 | Health Economics                            |
+-----+---------------------------------------------+
|  62 | Biostatistics                               |
+-----+---------------------------------------------+
|  67 | Quantitative Research Methods               |
+-----+---------------------------------------------+
|  69 | Urban Economics                             |
+-----+---------------------------------------------+
|  81 | Astronomy and Astrophysics                  |
+-----+---------------------------------------------+
|  84 | Quantum Physics                             |
+-----+---------------------------------------------+
|  85 | Thermodynamics                              |
+-----+---------------------------------------------+
|  86 | Fluid Mechanics                             |
+-----+---------------------------------------------+
|  87 | Solid State Physics                         |
+-----+---------------------------------------------+
|  88 | Classical Mechanics                         |
+-----+---------------------------------------------+
|  89 | Introduction to Civil Engineering           |
+-----+---------------------------------------------+
|  90 | Material Science and Engineering            |
+-----+---------------------------------------------+
|  91 | Structural Engineering                      |
+-----+---------------------------------------------+
|  92 | Environmental Engineering                   |
+-----+---------------------------------------------+
|  93 | Energy Systems Engineering                  |
+-----+---------------------------------------------+
|  94 | Aerodynamics                                |
+-----+---------------------------------------------+
|  95 | Heat Transfer                               |
+-----+---------------------------------------------+
|  96 | Renewable Energy Systems                    |
+-----+---------------------------------------------+
|  97 | Transportation Engineering                  |
+-----+---------------------------------------------+
| 102 | Business Analytics                          |
+-----+---------------------------------------------+

Output Statistics:

{'pos_cascade_threshold': 0.62, 'neg_cascade_threshold': 0.58, 'filters_resolved_by_helper_model': 101, 'filters_resolved_by_large_model': 2, 'num_routed_to_helper_model': 101}


Required Parameters
---------------------
- **user_instruction** : The user instruction for filtering.

Optional Parameters
----------------------
- **return_raw_outputs** : Whether to return raw outputs. Defaults to False.
- **default** : The default value for filtering in case of parsing errors. Defaults to True.
- **suffix** : The suffix for the new columns. Defaults to "_filter".
- **examples** : The examples dataframe. Defaults to None.
- **helper_examples** : The helper examples dataframe. Defaults to None.
- **strategy** : The reasoning strategy. Defaults to None.
- **cascade_args** : The arguments for join cascade. Defaults to None.
        recall_target : The target recall. Defaults to None.
        precision_target : The target precision when cascading. Defaults to None.
        sampling_percentage : The percentage of the data to sample when cascading. Defaults to 0.1.
        failure_probability : The failure probability when cascading. Defaults to 0.2.
- **return_stats** : Whether to return statistics. Defaults to False.