import os

import pandas as pd
import pytest

import lotus
from lotus.models import CrossEncoderReranker, LiteLLMRM, SentenceTransformersRM
from lotus.vector_store import FaissVS

################################################################################
# Setup
################################################################################
# Set logger level to DEBUG
lotus.logger.setLevel("DEBUG")

# Environment flags to enable/disable tests
ENABLE_OPENAI_TESTS = os.getenv("ENABLE_OPENAI_TESTS", "false").lower() == "true"
ENABLE_LOCAL_TESTS = os.getenv("ENABLE_LOCAL_TESTS", "false").lower() == "true"

# TODO: Add colbertv2 tests
MODEL_NAME_TO_ENABLED = {
    "intfloat/e5-small-v2": ENABLE_LOCAL_TESTS,
    "mixedbread-ai/mxbai-rerank-xsmall-v1": ENABLE_LOCAL_TESTS,
    "text-embedding-3-small": ENABLE_OPENAI_TESTS,
}
ENABLED_MODEL_NAMES = set([model_name for model_name, is_enabled in MODEL_NAME_TO_ENABLED.items() if is_enabled])

MODEL_NAME_TO_CLS = {
    "intfloat/e5-small-v2": SentenceTransformersRM,
    "mixedbread-ai/mxbai-rerank-xsmall-v1": CrossEncoderReranker,
    "text-embedding-3-small": LiteLLMRM,
}


def get_enabled(*candidate_models: str) -> list[str]:
    return [model for model in candidate_models if model in ENABLED_MODEL_NAMES]


@pytest.fixture(scope="session")
def setup_models():
    models = {}

    for model_name in ENABLED_MODEL_NAMES:
        models[model_name] = MODEL_NAME_TO_CLS[model_name](model=model_name)

    return models


################################################################################
# RM Only Tests
################################################################################
@pytest.mark.parametrize("model", get_enabled("intfloat/e5-small-v2", "text-embedding-3-small"))
def test_cluster_by(setup_models, model):
    rm = setup_models[model]
    vs = FaissVS()
    lotus.settings.configure(rm=rm, vs=vs)

    data = {
        "Course Name": [
            "Probability and Random Processes",
            "Cooking",
            "Food Sciences",
            "Optimization Methods in Engineering",
        ]
    }
    df = pd.DataFrame(data)
    df = df.sem_index("Course Name", "index_dir")
    df = df.sem_cluster_by("Course Name", 2)
    groups = df.groupby("cluster_id")["Course Name"].apply(set).to_dict()
    assert len(groups) == 2, groups
    if "Cooking" in groups[0]:
        cooking_group = groups[0]
        probability_group = groups[1]
    else:
        cooking_group = groups[1]
        probability_group = groups[0]

    assert cooking_group == {"Cooking", "Food Sciences"}, groups
    assert probability_group == {"Probability and Random Processes", "Optimization Methods in Engineering"}, groups


@pytest.mark.parametrize("model", get_enabled("intfloat/e5-small-v2", "text-embedding-3-small"))
def test_search_rm_only(setup_models, model):
    rm = setup_models[model]
    vs = FaissVS()

    lotus.settings.configure(rm=rm, vs=vs)

    data = {
        "Course Name": [
            "Probability and Random Processes",
            "Cooking",
            "Food Sciences",
            "Optimization Methods in Engineering",
        ]
    }
    df = pd.DataFrame(data)
    df = df.sem_index("Course Name", "index_dir")
    df = df.sem_search("Course Name", "Optimization", K=1)
    assert df["Course Name"].tolist() == ["Optimization Methods in Engineering"]


@pytest.mark.parametrize("model", get_enabled("intfloat/e5-small-v2", "text-embedding-3-small"))
def test_sim_join(setup_models, model):
    rm = setup_models[model]
    vs = FaissVS()
    lotus.settings.configure(rm=rm, vs=vs)

    data1 = {
        "Course Name": [
            "History of the Atlantic World",
            "Riemannian Geometry",
        ]
    }

    data2 = {"Skill": ["Math", "History"]}

    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2).sem_index("Skill", "index_dir")
    joined_df = df1.sem_sim_join(df2, left_on="Course Name", right_on="Skill", K=1)
    joined_pairs = set(zip(joined_df["Course Name"], joined_df["Skill"]))
    expected_pairs = {("History of the Atlantic World", "History"), ("Riemannian Geometry", "Math")}
    assert joined_pairs == expected_pairs, joined_pairs


# TODO: threshold is hardcoded for intfloat/e5-small-v2
@pytest.mark.skipif(
    "intfloat/e5-small-v2" not in ENABLED_MODEL_NAMES,
    reason="Skipping test because intfloat/e5-small-v2 is not enabled",
)
def test_dedup(setup_models):
    rm = setup_models["intfloat/e5-small-v2"]
    vs = FaissVS()
    lotus.settings.configure(rm=rm, vs=vs)
    data = {
        "Text": [
            "Probability and Random Processes",
            "Probability and Markov Chains",
            "Harry Potter",
            "Harry James Potter",
        ]
    }
    df = pd.DataFrame(data)
    df = df.sem_index("Text", "index_dir").sem_dedup("Text", threshold=0.85)
    kept = df["Text"].tolist()
    kept.sort()
    assert len(kept) == 2, kept
    assert "Harry" in kept[0], kept
    assert "Probability" in kept[1], kept


################################################################################
# VS Only Tests
################################################################################


@pytest.mark.parametrize("model", get_enabled("intfloat/e5-small-v2", "text-embedding-3-small"))
def test_vs_cluster_by(setup_models, model):
    rm = setup_models[model]
    vs = FaissVS()
    lotus.settings.configure(rm=rm, vs=vs)

    data = {
        "Course Name": [
            "Probability and Random Processes",
            "Cooking",
            "Food Sciences",
            "Optimization Methods in Engineering",
        ]
    }
    df = pd.DataFrame(data)
    df = df.sem_index("Course Name", "indexdir")
    df = df.sem_cluster_by("Course Name", 2)
    groups = df.groupby("cluster_id")["Course Name"].apply(set).to_dict()
    assert len(groups) == 2, groups
    if "Cooking" in groups[0]:
        cooking_group = groups[0]
        probability_group = groups[1]
    else:
        cooking_group = groups[1]
        probability_group = groups[0]

    assert cooking_group == {"Cooking", "Food Sciences"}, groups
    assert probability_group == {"Probability and Random Processes", "Optimization Methods in Engineering"}, groups


@pytest.mark.parametrize("model", get_enabled("intfloat/e5-small-v2", "text-embedding-3-small"))
def test_vs_search_rm_only(setup_models, model):
    rm = setup_models[model]
    vs = FaissVS()
    lotus.settings.configure(rm=rm, vs=vs)

    data = {
        "Course Name": [
            "Probability and Random Processes",
            "Cooking",
            "Food Sciences",
            "Optimization Methods in Engineering",
        ]
    }
    df = pd.DataFrame(data)
    df = df.sem_index("Course Name", "secondindexdir")
    df = df.sem_search("Course Name", "Optimization", K=1)
    assert df["Course Name"].tolist() == ["Optimization Methods in Engineering"]


@pytest.mark.parametrize("model", get_enabled("intfloat/e5-small-v2", "text-embedding-3-small"))
def test_vs_sim_join(setup_models, model):
    rm = setup_models[model]
    vs = FaissVS()
    lotus.settings.configure(rm=rm, vs=vs)

    data1 = {
        "Course Name": [
            "History of the Atlantic World",
            "Riemannian Geometry",
        ]
    }

    data2 = {"Skill": ["Math", "History"]}

    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2).sem_index("Skill", "thirdindexdir")
    joined_df = df1.sem_sim_join(df2, left_on="Course Name", right_on="Skill", K=1)
    joined_pairs = set(zip(joined_df["Course Name"], joined_df["Skill"]))
    expected_pairs = {("History of the Atlantic World", "History"), ("Riemannian Geometry", "Math")}
    assert joined_pairs == expected_pairs, joined_pairs


# TODO: threshold is hardcoded for intfloat/e5-small-v2
@pytest.mark.skipif(
    "intfloat/e5-small-v2" not in ENABLED_MODEL_NAMES,
    reason="Skipping test because intfloat/e5-small-v2 is not enabled",
)
def test_vs_dedup(setup_models):
    rm = setup_models["intfloat/e5-small-v2"]
    vs = FaissVS()
    lotus.settings.configure(rm=rm, vs=vs)
    data = {
        "Text": [
            "Probability and Random Processes",
            "Probability and Markov Chains",
            "Harry Potter",
            "Harry James Potter",
        ]
    }
    df = pd.DataFrame(data)
    df = df.sem_index("Text", "fourthindexdir").sem_dedup("Text", threshold=0.85)
    kept = df["Text"].tolist()
    kept.sort()
    assert len(kept) == 2, kept
    assert "Harry" in kept[0], kept
    assert "Probability" in kept[1], kept


################################################################################
# Reranker Only Tests
################################################################################
@pytest.mark.parametrize("model", get_enabled("mixedbread-ai/mxbai-rerank-xsmall-v1"))
def test_search_reranker_only(setup_models, model):
    reranker = setup_models[model]
    lotus.settings.configure(reranker=reranker)

    data = {
        "Course Name": [
            "Probability and Random Processes",
            "Cooking",
            "Food Sciences",
            "Optimization Methods in Engineering",
        ]
    }
    df = pd.DataFrame(data)
    df = df.sem_search("Course Name", "Optimization", n_rerank=2)
    assert df["Course Name"].tolist() == ["Optimization Methods in Engineering", "Probability and Random Processes"]


################################################################################
# Combined Tests
################################################################################
# TODO: Figure out how to parameterize pairs of models
@pytest.mark.skipif(not ENABLE_LOCAL_TESTS, reason="Skipping test because local tests are not enabled")
def test_search(setup_models):
    models = setup_models
    rm = models["intfloat/e5-small-v2"]
    vs = FaissVS()
    reranker = models["mixedbread-ai/mxbai-rerank-xsmall-v1"]
    lotus.settings.configure(rm=rm, vs=vs, reranker=reranker)

    data = {
        "Course Name": [
            "Probability and Random Processes",
            "Cooking",
            "Food Sciences",
            "Optimization Methods in Engineering",
        ]
    }
    df = pd.DataFrame(data)
    df = df.sem_index("Course Name", "index_dir")
    df = df.sem_search("Course Name", "Optimization", K=2, n_rerank=1)
    assert df["Course Name"].tolist() == ["Optimization Methods in Engineering"]


@pytest.mark.parametrize("model", get_enabled("intfloat/e5-small-v2", "text-embedding-3-small"))
def test_filtered_vector_search(setup_models, model):
    """
    Test filtered vector search.

    This test starts with a DataFrame that contains:
      - a text column ("Course Name") that will be embedded and indexed,
      - a structured column ("Category") used for filtering.

    The test performs the following steps:
      1. Index the "Course Name" column to generate semantic embeddings.
      2. Apply a filter to keep only rows with Category "Culinary".
      3. Perform a semantic search over the filtered DataFrame using a query ("advanced") that is
         expected to pick out the culinary course "Gourmet Cooking Advanced".
    """
    rm = setup_models[model]
    vs = FaissVS()
    lotus.settings.configure(rm=rm, vs=vs)

    data = {
        "Course Name": [
            "Gourmet Cooking Advanced",
            "Home Cooking Basics",
            "Probability and Statistics",
            "Linear Algebra Fundamentals",
        ],
        "Category": ["Culinary", "Culinary", "Math", "Math"],
    }
    df = pd.DataFrame(data)
    # Index the 'Course Name' column to generate semantic embeddings.
    df = df.sem_index("Course Name", "filtered_index_dir")
    # Filter the DataFrame to only include Culinary courses.
    df_filtered = df[df["Category"] == "Culinary"]
    # Perform semantic search on the filtered DataFrame.
    df_searched = df_filtered.sem_search("Course Name", "advanced", K=1)

    # Verify that every returned row belongs to the Culinary category.
    assert all(df_searched["Category"] == "Culinary"), "Filtered search returned non-Culinary courses."

    # Verify the expected course is returned.
    expected_course = "Gourmet Cooking Advanced"
    result_course = df_searched["Course Name"].iloc[0]
    assert result_course == expected_course, f"Expected '{expected_course}', but got '{result_course}'"
