import pandas as pd
import weaviate

import lotus
from lotus.models import SentenceTransformersRM
from lotus.vector_store import WeaviateVS

# First run `docker run -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:1.29.1` to start the weaviate server
client = weaviate.connect_to_local()

rm = SentenceTransformersRM(model="intfloat/e5-base-v2")
vs = WeaviateVS(client)

lotus.settings.configure(rm=rm, vs=vs)
data = {
    "Course Name": [
        "Probability and Random Processes",
        "Optimization Methods in Engineering",
        "Digital Design and Integrated Circuits",
        "Computer Security",
        "Introduction to Computer Science",
        "Introduction to Data Science",
        "Introduction to Machine Learning",
        "Introduction to Artificial Intelligence",
        "Introduction to Robotics",
        "Introduction to Computer Vision",
        "Introduction to Natural Language Processing",
        "Introduction to Reinforcement Learning",
        "Introduction to Deep Learning",
        "Introduction to Computer Networks",
    ]
}
df = pd.DataFrame(data)

df = df.sem_index("Course Name", "index_dir").sem_search(
    "Course Name",
    "Which course name is most related to machine learning?",
    K=8,
)
print(df)
