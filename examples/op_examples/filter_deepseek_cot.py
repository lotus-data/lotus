import pandas as pd

import lotus
from lotus.models import LM
from lotus.sem_ops.postprocessors import deepseek_cot_postprocessor

lm = LM(model="ollama/deepseek-r1:7b")

lotus.settings.configure(lm=lm)

data = {
    "Reviews": [
        "I absolutely love this product. It exceeded all my expectations.",
        "Terrible experience. The product broke within a week.",
        "The quality is average, nothing special.",
        "Fantastic service and high quality!",
        "I would not recommend this to anyone.",
    ]
}
df = pd.DataFrame(data)
user_instruction = "{Reviews} are positive reviews"
df = df.sem_filter(
    user_instruction, return_explanations=True, return_all=True, reasoning_parser=deepseek_cot_postprocessor
)
print(df)
