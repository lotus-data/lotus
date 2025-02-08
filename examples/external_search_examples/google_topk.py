import lotus
from lotus import ExternalSearchCorpus, sem_external_search
from lotus.models import LM

lm = LM(model="gpt-4o-mini")

lotus.settings.configure(lm=lm)

df = sem_external_search(ExternalSearchCorpus.GOOGLE, "deep learning", 5)[["title", "snippet"]]
print(f"Results from Google\n{df}")
most_interesting_articles = df.sem_topk("Which {snippet} is most interesting?", K=1)
print(f"Most interesting articles\n{most_interesting_articles}")
