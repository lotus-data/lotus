import pandas as pd

import lotus
from lotus.models import LM
from lotus.types import ReasoningStrategy

lm = LM(model="gpt-4o-mini")
lotus.settings.configure(lm=lm)

# Create a more complex dataset for extraction with reasoning
df = pd.DataFrame(
    {
        "movie_description": [
            "The Dark Knight is a 2008 superhero film directed by Christopher Nolan. It stars Christian Bale as Batman and Heath Ledger as the Joker.",
            "Inception, released in 2010, is a sci-fi thriller directed by Christopher Nolan starring Leonardo DiCaprio and Marion Cotillard.",
            "Pulp Fiction is a 1994 crime film written and directed by Quentin Tarantino, featuring John Travolta and Samuel L. Jackson.",
            "The Godfather (1972) is an American crime film directed by Francis Ford Coppola, starring Marlon Brando and Al Pacino.",
            "Casablanca is a 1942 romantic drama film directed by Michael Curtiz, starring Humphrey Bogart and Ingrid Bergman.",
        ]
    }
)

input_cols = ["movie_description"]

# Extract multiple attributes with reasoning
output_cols = {
    "title": "The title of the movie",
    "director": "The director of the movie",
    "year": "The release year of the movie",
    "genre": "The primary genre of the movie",
    "lead_actor": "The main actor mentioned first",
}

print("=== Extract with Chain of Thought Reasoning ===")
# Use CoT reasoning to show the model's thought process
new_df = df.sem_extract(
    input_cols, output_cols, strategy=ReasoningStrategy.ZS_COT, return_explanations=True, extract_quotes=False
)

print(new_df[["title", "director", "year", "genre", "lead_actor"]])

print("\n=== Explanations (Reasoning Chains) ===")
for i, explanation in enumerate(new_df["explanation"]):
    print(f"\nRow {i} - {new_df.loc[i, 'title']}:")
    print(f"Reasoning: {explanation}")

print("\n=== Extract with Quotes and CoT Reasoning ===")
# Also demonstrate with quotes extraction
new_df_with_quotes = df.sem_extract(
    input_cols,
    {"director": "The director of the movie"},
    strategy=ReasoningStrategy.ZS_COT,
    return_explanations=True,
    extract_quotes=True,
)

print(new_df_with_quotes[["director", "director_quote", "explanation"]])
