import pandas as pd

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
        "Gradient descent is an optimization algorithm that minimizes cost functions by iteratively moving in the direction of steepest descent of the gradient.",
        "Cross-validation helps assess model performance by splitting data into training and validation sets multiple times to get a better estimate of how the model generalizes.",
        "Ensemble methods combine multiple models to improve performance. They reduce overfitting and variance, often leading to better generalization than individual models.",
    ],
}

df = pd.DataFrame(data)
judge_instruction = "Rate the accuracy and completeness of this {answer} to the {question} on a scale of 1-10, where 10 is excellent. Only output the score."

results = df.llm_as_judge(
    judge_instruction=judge_instruction,
    n_trials=2,
)

print(results)
