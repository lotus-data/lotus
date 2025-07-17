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

# Basic example - single output per row
print("\n===== Basic Example - Single Output =====")
user_instruction = "What is a similar course to {Course Name}. Be concise."
df_basic = df.sem_map(user_instruction)
print(df_basic)

# Example with multiple samples - generate 3 alternatives per course
print("\n===== Multiple Samples Example =====")
user_instruction = "Suggest an alternative course to {Course Name}. Be creative."
df_multi = df.sem_map(
    user_instruction,
    nsample=3,     # Generate 3 alternatives per course
    temp=0.7,      # Higher temperature for more varied outputs
)
print(df_multi)

# Example with temperature but single sample
print("\n===== Temperature Example (Higher Creativity) =====")
user_instruction = "If {Course Name} was a book title, what would it be called? Be creative."
df_temp = df.sem_map(
    user_instruction,
    temp=1.0,      # High temperature for maximum creativity
)
print(df_temp)
