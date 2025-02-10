sem_join
=================

Overview
----------
The sem_join operator in joins to datasets according to the langex, which specifies a predicate in natural language. 

Motivation
-----------
Traditional join operations often rely on rigid equality conditions, making them unsuitable for scenarios requiring nuanced, 
context-aware relationships. The sem_join operator addresses these limitations by enabling semantic matching of rows between 
datasets based on natural language predicates


Join Example
--------------
.. code-block:: python

    import pandas as pd

    import lotus
    from lotus.models import LM

    lm = LM(model="gpt-4o-mini")

    lotus.settings.configure(lm=lm)
    data = {
        "Course Name": [
            "History of the Atlantic World",
            "Riemannian Geometry",
            "Operating Systems",
            "Food Science",
            "Compilers",
            "Intro to computer science",
        ]
    }

    data2 = {"Skill": ["Math", "Computer Science"]}

    df1 = pd.DataFrame(data)
    df2 = pd.DataFrame(data2)
    join_instruction = "Taking {Course Name:left} will help me learn {Skill:right}"
    res = df1.sem_join(df2, join_instruction)
    print(res)

Output:

+---+----------------------------+-------------------+
|   |      Course Name           |       Skill       |
+---+----------------------------+-------------------+                
| 1 |  Riemannian Geometry       |       Math        |
+---+----------------------------+-------------------+
| 2 |   Operating Systems        |  Computer Science |
+---+----------------------------+-------------------+
| 4 |      Compilers             |  Computer Science |
+---+----------------------------+-------------------+
| 5 | Intro to computer science  |  Computer Science |
+---+----------------------------+-------------------+



Example of Join with Approximation
----------------------
.. code-block:: python

    import pandas as pd

    import lotus
    from lotus.models import LM, SentenceTransformersRM
    from lotus.types import CascadeArgs

    lm = LM(model="gpt-4o-mini")
    rm = SentenceTransformersRM(model="intfloat/e5-base-v2")

    lotus.settings.configure(lm=lm, rm=rm)
    data = {
        "Course Name": [
            "Digital Design and Integrated Circuits",
            "Data Structures and Algorithms",
            "The History of Art",
            "Natural Language Processing",
        ]
    }

    skills = [
        "Math", "Computer Science", "Management", "Creative Writing", "Data Analysis", "Machine Learning", "Project Management",
        "Problem Solving", "Singing", "Critical Thinking", "Public Speaking", "Teamwork", "Adaptability", "Programming",
        "Leadership", "Time Management", "Negotiation", "Decision Making", "Networking", "Painting",
        "Customer Service", "Marketing", "Graphic Design", "Nursery", "SEO", "Content Creation", "Video Editing", "Sales",
        "Financial Analysis", "Accounting", "Event Planning", "Foreign Languages", "Software Development", "Cybersecurity",
        "Social Media Management", "Photography", "Writing & Editing", "Technical Support", "Database Management", "Web Development",
        "Business Strategy", "Operations Management", "UI/UX Design", "Reinforcement Learning", "Data Visualization",
        "Product Management", "Cloud Computing", "Agile Methodology", "Blockchain", "IT Support", "Legal Research", "Supply Chain Management",
        "Copywriting", "Human Resources", "Quality Assurance", "Medical Research", "Healthcare Management", "Sports Coaching",
        "Editing & Proofreading", "Legal Writing", "Human Anatomy", "Chemistry", "Physics", "Biology",
        "Psychology", "Sociology", "Anthropology", "Political Science", "Public Relations", "Fashion Design", "Interior Design",
        "Automotive Repair", "Plumbing", "Carpentry", "Electrical Work", "Welding", "Electronics", "Hardware Engineering",
        "Circuit Design", "Robotics", "Environmental Science", "Marine Biology", "Urban Planning", "Geography",
        "Agricultural Science", "Animal Care", "Veterinary Science", "Zoology", "Ecology", "Botany", "Landscape Design",
        "Baking & Pastry", "Culinary Arts", "Bartending", "Nutrition", "Dietary Planning", "Physical Training", "Yoga",
    ]
    data2 = pd.DataFrame({"Skill": skills})


    df1 = pd.DataFrame(data)
    df2 = pd.DataFrame(data2)
    join_instruction = "By taking {Course Name:left} I will learn {Skill:right}"

    cascade_args = CascadeArgs(recall_target=0.7, precision_target=0.7)
    res, stats = df1.sem_join(df2, join_instruction, cascade_args=cascade_args, return_stats=True)


    print(f"Joined {df1.shape[0]} rows from df1 with {df2.shape[0]} rows from df2")
    print(f"    Join cascade took {stats['join_resolved_by_large_model']} LM calls")
    print(f"    Helper resolved {stats['join_resolved_by_helper_model']} LM calls")
    print(f"Join cascade used {stats['total_LM_calls']} LM calls in total")
    print(f"Naive join would require {df1.shape[0]*df2.shape[0]} LM calls")
    print(res)

Output:

+---+----------------------------------------+----------------------+
|   |            Course Name                 |        Skill         |
+---+----------------------------------------+----------------------+
| 0 | Digital Design and Integrated Circuits | Circuit Design       |
+---+----------------------------------------+----------------------+
| 3 | Natural Language Processing            | Machine Learning     |
+---+----------------------------------------+----------------------+
| 1 | Data Structures and Algorithms         | Computer Science     |
+---+----------------------------------------+----------------------+
| 0 | Digital Design and Integrated Circuits | Electronics          |
+---+----------------------------------------+----------------------+
| 0 | Digital Design and Integrated Circuits | Hardware Engineering |
+---+----------------------------------------+----------------------+


Required Parameters
----------------------
- **other** : The other dataframe or series to join with.
- **join_instruction** : The user instruction for join.

Optional Parameters
----------------------
- **return_explanations** : Whether to return explanations. Defaults to False.
- **how** : The type of join to perform. Defaults to "inner".
- **suffix** : The suffix for the new columns. Defaults to "_join".
- **examples** : The examples dataframe. Defaults to None.
- **strategy** : The reasoning strategy. Defaults to None.
- **default** : The default value for the join in case of parsing errors. Defaults to True.
- **cascade_args**: The arguments for join cascade. Defaults to None.
    recall_target : The target recall. Defaults to None.
    precision_target : The target precision when cascading. Defaults to None.
    sampling_percentage : The percentage of the data to sample when cascading. Defaults to 0.1.
    failure_probability : The failure probability when cascading. Defaults to 0.2.
    map_instruction : The map instruction when cascading. Defaults to None.
    map_examples : The map examples when cascading. Defaults to None.
- **return_stats** : Whether to return stats. Defaults to False.