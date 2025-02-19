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
df = df.sem_filter(user_instruction, strategy="cot")
print(df)
