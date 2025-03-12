import lotus
from lotus import WebSearchCorpus, web_search
from lotus.models import LM

lm = LM(model="gpt-4o-mini")

lotus.settings.configure(lm=lm)

df = web_search(WebSearchCorpus.BING, "state-of-the-art AI models", 10)[["title", "snippet"]]
print(f"Results from Bing:\n{df}\n")

top_bing_articles = df.sem_topk("Which {snippet} provides the best insight into AI models?", K=3)
print(f"Top 3 most insightful articles from Bing:\n{top_bing_articles}")
