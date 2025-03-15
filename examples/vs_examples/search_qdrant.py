import pandas as pd
from qdrant_client import QdrantClient

import lotus
from lotus.models import SentenceTransformersRM
from lotus.vector_store import QdrantVS

# Run this command to start the qdrant server
# docker run -p 6333:6333 -p 6334:6334 \
#    -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
#    qdrant/qdrant
client = QdrantClient(url="http://localhost:6333")
rm = SentenceTransformersRM(model="intfloat/e5-base-v2")
vs = QdrantVS(client)

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
