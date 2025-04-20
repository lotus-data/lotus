import pandas as pd

import lotus
from lotus.models import LM

lm = LM(model="gpt-4o-mini")

lotus.settings.configure(lm=lm)

# Creating a sample DataFrame
data = {
    "title": [
        "Who's Who: Large Language Models Meet Knowledge",
        "Using GPT Models for Qualitative and Quantitative Analysis",
        "Transit Pulse: Utilizing Social Media as a Source",
        "A Distribution Semantics for Probabilistic Term Rewriting",
        "Natural Language Querying System Through Entity Recognition",
        "Performance-Driven QUBO for Recommender Systems",
        "From Tokens to Materials: Leveraging Language Models",
        "A Recommendation Model Utilizing Separation Embeddings",
        "Incorporating Group Prior into Variational Inference",
        "HyQE: Ranking Contexts with Hybrid Query Expansion",
    ],
    "authors": [
        ["Quang Hieu Pham", "Hoang Ngo", "Anh Tuan Lu"],
        ["Bohdan M. Pavlyshenko"],
        ["Jiahao Wang", "Amer Shalaby"],
        ["Germán Vidal"],
        ["Joshua Amavi", "Mirian Halfeld Ferrari", "Nicolas Schmit"],
        ["Jiayang Niu", "Jie Li", "Ke Deng", "Mark San"],
        ["Yuwei Wan", "Tong Xie", "Nan Wu", "Wenjie Zhou"],
        ["Wenyi Liu", "Rui Wang", "Yuanshuai Luo", "Jian Sun"],
        ["Han Xu", "Taoxing Pan", "Zhiqiang Liu", "Xia Wu"],
        ["Weichao Zhou", "Jiaxin Zhang", "Hilat Hasson"],
    ],
    "abstract": [
        "Retrieval-augmented generation (RAG) methods are becoming increasingly popular...",
        "The paper considers an approach of using Google’s GPT models for market analysis...",
        "Users of the transit system flood social networks with posts...",
        "Probabilistic programming is becoming increasingly relevant...",
        "This paper focuses on a domain expert querying large knowledge graphs...",
        "We propose Counterfactual Analysis Quadratic Unconstrained Binary Optimization...",
        "Exploring the predictive capabilities of language models in material discovery...",
        "With the explosive growth of Internet data, users require personalized recommendation...",
        "User behavior modeling – which aims to extract latent features...",
        "In retrieval-augmented systems, ranking relevant contexts is a challenge...",
    ],
    "arxiv_id": [
        "2410.15737",
        "2410.15884",
        "2410.15016",
        "2410.15081",
        "2410.15753",
        "2410.15272",
        "2410.16165",
        "2410.15026",
        "2410.15098",
        "2410.15262",
    ],
}

# Create DataFrame
df = pd.DataFrame(data)
filtered_df = df.iloc[2:7]

input_cols = ["abstract"]
output_cols = {"topics": None}

new_df1 = filtered_df.sem_extract(input_cols, output_cols)
new_df2 = df.sem_extract(input_cols, output_cols)
print(new_df1)
print(new_df2)
