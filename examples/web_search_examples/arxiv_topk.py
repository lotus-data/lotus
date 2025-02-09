import lotus
from lotus import WebSearchCorpus, web_search
from lotus.models import LM

lm = LM(model="gpt-4o-mini")

lotus.settings.configure(lm=lm)

df = web_search(WebSearchCorpus.ARXIV, "deep learning", 5)[["title", "abstract"]]
print(f"Results from Arxiv\n{df}")
most_interesting_articles = df.sem_topk("Which {abstract} is most interesting?", K=1)
print(f"Most interesting articles\n{most_interesting_articles}")
