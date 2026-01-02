import pandas as pd
from crewai_tools import SerperDevTool, WebsiteSearchTool

import lotus
from lotus.models import LM

lm = LM()

lotus.settings.configure(lm=lm)

dept = {
    "University Department": [
        "Stanford University Computer Science Department",
        "Berkeley EECS Department",
    ]
}

query = """
Find the hyperlinks of Reasearch Labs associated with this {University Department}. Return the hyperlinks as a list.
"""
df = pd.DataFrame(dept)
result = df.sem_map(query, tools=[SerperDevTool(), WebsiteSearchTool()])
print(result)


people = {
    "Person Name": [
        "Roger Federer",
        "Cristiano Ronaldo",
    ]
}

df2 = pd.DataFrame(people)
user_instruction2 = "Find recent popular news articles on {Person Name} and summarize the main points in 2-3 sentences."
result2 = df2.sem_map(user_instruction2, tools=[SerperDevTool(), WebsiteSearchTool()])
print(result2)
