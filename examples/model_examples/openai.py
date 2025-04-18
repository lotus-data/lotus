import pandas as pd

import lotus
from lotus.models import LM

# Set up model
lm = LM(model="gpt-4o-mini")
lotus.settings.configure(lm=lm)

# Input data: Research paper titles
data = {
    "Research Title": [
        "Using Graph Neural Networks for Protein Structure Prediction",
        "Transformer-based Models for Machine Translation",
        "A Novel Approach to Quantum Error Correction Codes",
        "Deep Reinforcement Learning for Robotic Arm Control",
    ]
}

df = pd.DataFrame(data)

# Instruction: Find a semantically similar research topic
user_instruction = "Suggest a similar research topic to {Research Title}. Be concise."

df = df.sem_map(user_instruction)
print(df)
