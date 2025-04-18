import pandas as pd

import lotus
from lotus.models import LM

# Set up model with llama3
lm = LM(model="ollama/llama3.2")
lotus.settings.configure(lm=lm)

# Input data: customer-reported issues
data = {
    "Issue Description": [
        "My wireless earbuds keep disconnecting when I move more than 10 feet from my phone.",
        "The smart thermostat doesn't respond to voice commands through Alexa.",
        "Laptop battery drains from 100% to 30% in less than an hour even when idle.",
        "My robot vacuum maps the room but refuses to clean certain sections.",
    ]
}

df = pd.DataFrame(data)

# Semantic instruction: suggest similar problem
user_instruction = "Suggest a similar technical issue to: {Issue Description}. Keep it short."

# Apply sem_map
df = df.sem_map(user_instruction)

print(df)
