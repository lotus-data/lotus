import logging

import pandas as pd
from pydantic import BaseModel

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
user_instruction = (
    "What is a similar course to {Course Name}. Also give a study plan for the similar course. Be concise."
)


class Course(BaseModel):
    new_course_name: str
    new_course_study_plan: str


df = df.sem_map(user_instruction, response_format=Course)
print(df)
