import pandas as pd
from pydantic import BaseModel, Field

import lotus
from lotus.models import LM

# Configure the language model
lm = LM(model="gpt-4o-mini")
lotus.settings.configure(lm=lm)

# Sample data representing student responses to evaluate
data = {
    "student_id": [1, 2, 3, 4],
    "question": [
        "Explain the difference between supervised and unsupervised learning",
        "What is the purpose of cross-validation in machine learning?",
        "Describe how gradient descent works",
        "What are the advantages of ensemble methods?",
    ],
    "answer": [
        "Supervised learning uses labeled data to train models, while unsupervised learning finds patterns in unlabeled data. For example, classification is supervised, clustering is unsupervised.",
        "Cross-validation helps assess model performance by splitting data into training and validation sets multiple times to get a better estimate of how the model generalizes.",
        "Gradient descent is an optimization algorithm that minimizes cost functions by iteratively moving in the direction of steepest descent of the gradient.",
        "Ensemble methods combine multiple models to improve performance. They reduce overfitting and variance, often leading to better generalization than individual models.",
    ],
}
df = pd.DataFrame(data)


class EvaluationScore(BaseModel):
    score: int = Field(description="Score from 1-10")
    reasoning: str = Field(description="Detailed reasoning for the score")
    strengths: list[str] = Field(description="Key strengths of the answer")
    improvements: list[str] = Field(description="Areas for improvement")


results = df.llm_as_judge(
    judge_instruction="Evaluate the student {answer} for the {question}",
    response_format=EvaluationScore,
    suffix="_evaluation",
)
print(results)
