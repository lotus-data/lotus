import pandas as pd
import pytest
from qdrant_client import QdrantClient

import lotus
from lotus.models import LiteLLMRM
from lotus.vector_store import QdrantVS
from tests.base_test import BaseTest


@pytest.fixture(scope="module")
def qdrant_vs():
    client = QdrantClient(url="http://localhost:6333")
    vs = QdrantVS(client)
    return vs


@pytest.fixture(scope="module")
def rm():
    return LiteLLMRM(model="text-embedding-3-small")


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "Course Name": [
                "Introduction to Cooking",
                "Advanced Cooking",
                "Basic Mathematics",
                "Advanced Mathematics",
                "Machine Learning 101",
                "Deep Learning for Beginners",
                "History of Art",
                "Modern Art Techniques",
            ],
            "Department": ["Culinary", "Culinary", "Math", "Math", "CS", "CS", "Art", "Art"],
            "Level": [100, 200, 100, 200, 300, 400, 100, 200],
        }
    )


class TestQdrantVS(BaseTest):
    @pytest.fixture(autouse=True)
    def setup_vs(self, rm, qdrant_vs):
        lotus.settings.configure(rm=rm, vs=qdrant_vs)

    def test_index_creation_and_reload(self, sample_df):
        df = sample_df.copy()
        df = df.sem_index("Course Name", "qdrant_test_index")
        # Now reload the index and check it works
        lotus.settings.vs.load_index("qdrant_test_index")
        # Should not raise
        assert lotus.settings.vs.index_dir == "qdrant_test_index"

    def test_simple_sem_search(self, sample_df):
        df = sample_df.copy().sem_index("Course Name", "qdrant_search_index")
        result = df.sem_search("Course Name", "Find the course about machine learning", K=1)
        assert len(result) == 1
        assert result["Course Name"].iloc[0] == "Machine Learning 101"

    def test_sem_search_with_filter(self, sample_df):
        df = sample_df.copy().sem_index("Course Name", "qdrant_filter_index")
        filtered_df = df[df["Department"] == "CS"]
        result = filtered_df.sem_search("Course Name", "Find the course about deep learning", K=1)
        assert len(result) == 1
        assert result["Course Name"].iloc[0] == "Deep Learning for Beginners"
        assert all(dept == "CS" for dept in result["Department"])

    def test_sem_join(self, sample_df):
        left = pd.DataFrame({"Skill": ["Machine Learning", "Cooking"]})
        right = sample_df.copy().sem_index("Course Name", "qdrant_join_index")
        joined = left.sem_sim_join(right, left_on="Skill", right_on="Course Name", K=1)
        assert len(joined) == 2
        # Each skill should match the most obvious course(s)
        expected = {
            "Machine Learning": ["Machine Learning 101"],
            "Cooking": ["Introduction to Cooking", "Advanced Cooking"],
        }
        for _, row in joined.iterrows():
            assert any(exp in row["Course Name"] for exp in expected[row["Skill"]])
