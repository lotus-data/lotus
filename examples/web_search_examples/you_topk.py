import lotus
from lotus import WebSearchCorpus, web_search
from lotus.models import LM

lm = LM(model="gpt-4o-mini")

lotus.settings.configure(lm=lm)

df = web_search(WebSearchCorpus.YOU, "latest AI breakthroughs", 10)[["title", "snippet"]]
print(f"Results from You.com:\n{df}\n")

top_you_articles = df.sem_topk("Which {snippet} is the most groundbreaking?", K=3)
print(f"Top 3 most interesting articles from You.com:\n{top_you_articles}")
