import lotus
from lotus import WebSearchCorpus, web_extract
from lotus.models import LM

lm = LM(model="gpt-4o-mini")

lotus.settings.configure(lm=lm)

# Extract full text from an arXiv paper using its ID
df = web_extract(WebSearchCorpus.ARXIV, doc_id="2303.08774")
print(f"Extracted from ArXiv:\n{df}\n\n")

# Use the extracted full text for semantic operations
if df["full_text"].iloc[0]:
    print(f"Full text length: {len(df['full_text'].iloc[0])} characters")
    print(f"First 500 characters:\n{df['full_text'].iloc[0][:500]}...")
