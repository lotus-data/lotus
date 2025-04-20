import pandas as pd

import lotus
from lotus.models import LM, SentenceTransformersRM
from lotus.vector_store import FaissVS

lm = LM(model="gpt-4o-mini")
rm = SentenceTransformersRM(model="intfloat/e5-base-v2")
vs = FaissVS()

lotus.settings.configure(lm=lm, rm=rm, vs=vs)
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
df = df.sem_index("Course Name", "course_name_index").sem_cluster_by("Course Name", 2)
print(df)
