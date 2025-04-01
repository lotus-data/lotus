import pandas as pd

import lotus
from lotus.models import LM, LiteLLMRM
from lotus.vector_store import FaissVS

lm = LM(model="gpt-4o-mini")
rm = LiteLLMRM(model="text-embedding-3-small")
vs = FaissVS()

lotus.settings.configure(lm=lm, rm=rm, vs=vs)
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
df2 = pd.DataFrame(data2).sem_index("Skill", "skill_index")
res = df1.sem_sim_join(df2, left_on="Course Name", right_on="Skill", K=1)
print(res)
