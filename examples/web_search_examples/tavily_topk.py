import lotus
from lotus import WebSearchCorpus, web_search
from lotus.models import LM

lm = LM(model="gpt-4o-mini")

lotus.settings.configure(lm=lm)

df = web_search(WebSearchCorpus.TAVILY, "AI ethics in 2025", 10)[["title", "summary"]]
print(f"Results from Tavily:\n{df}\n")

top_tavily_articles = df.sem_topk("Which {summary} best explains ethical concerns in AI?", K=3)
print(f"Top 3 articles from Tavily on AI ethics:\n{top_tavily_articles}")
