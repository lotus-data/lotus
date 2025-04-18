import os

import pandas as pd

import lotus
from lotus.cache import CacheConfig, CacheFactory, CacheType
from lotus.models import LM

cache_config = CacheConfig(cache_type=CacheType.SQLITE, max_size=1000, cache_dir=os.path.expanduser("~/.lotus/cache"))
cache = CacheFactory.create_cache(cache_config)

lm = LM(model="gpt-4o-mini", cache=cache)

lotus.settings.configure(lm=lm, enable_cache=True)  # default caching is False
data = {
    "Course Name": [
        "Probability and Random Processes",
        "Optimization Methods in Engineering",
        "Digital Design and Integrated Circuits",
        "Computer Security",
    ]
}
df = pd.DataFrame(data)
user_instruction = "{Course Name} requires a lot of math"
df = df.sem_filter(user_instruction)
print("====== intial run ======")
print(df)
lm.print_total_usage()

# run a second time
df = df.sem_filter(user_instruction)
print("====== second run ======")
print(df)
lm.print_total_usage()
