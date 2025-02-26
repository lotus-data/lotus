import pandas as pd
from pydantic import BaseModel

import lotus
from lotus.models import LM

lm = LM(model="gpt-4o-mini", structured_format=True)

lotus.settings.configure(lm=lm)
data = {
    "Course Name": [
        "Probability and Random Processes",
        "Optimization Methods in Engineering",
        "Digital Design and Integrated Circuits",
        "Computer Security",
    ]
}


class Test(BaseModel):
    Answer: str
    Reasoning: str


df = pd.DataFrame(data)
user_instruction = "What is a similar course to {Course Name}. Be concise."
df = df.sem_map(user_instruction, response_format=Test)
print(df)
