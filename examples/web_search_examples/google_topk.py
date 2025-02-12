import lotus
from lotus import WebSearchCorpus, web_search
from lotus.models import LM

lm = LM(model="gpt-4o-mini")

lotus.settings.configure(lm=lm)

df = web_search(WebSearchCorpus.GOOGLE, "deep learning research", 5)[["title", "snippet"]]
print(f"Results from Google\n{df}")
most_interesting_articles = df.sem_topk("Which {snippet} is the most exciting?", K=1)
print(f"Most interesting articles\n{most_interesting_articles}")
