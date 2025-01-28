import os

import pandas as pd
import pytest

import lotus
from lotus.models import CrossEncoderReranker, LiteLLMRM, SentenceTransformersRM
from lotus.vector_store import ChromaVS, FaissVS, PineconeVS, QdrantVS, WeaviateVS

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

VECTOR_STORE_TO_CLS = {
    'local': FaissVS,
    'weaviate':WeaviateVS,
    'pinecone': PineconeVS,
    'chroma': ChromaVS,
    'qdrant': QdrantVS
}


def get_enabled(*candidate_models: str) -> list[str]:
    return [model for model in candidate_models if model in ENABLED_MODEL_NAMES]


@pytest.fixture(scope="session")
def setup_models():
    models = {}

    for model_name in ENABLED_MODEL_NAMES:
        models[model_name] = MODEL_NAME_TO_CLS[model_name](model=model_name)


    return models


@pytest.fixture(scope='session')
def setup_vs():
    vs_model = {}

    for vs in VECTOR_STORE_TO_CLS:
        vs_model[vs] = VECTOR_STORE_TO_CLS[vs]()

    return vs_model

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
    lotus.settings.configure(rm=rm,vs=vs)
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


@pytest.mark.parametrize("vs", VECTOR_STORE_TO_CLS.keys())
@pytest.mark.parametrize("model", get_enabled("intfloat/e5-small-v2", "text-embedding-3-small"))
def test_vs_cluster_by(setup_models, setup_vs, vs, model):
    rm = setup_models[model]
    my_vs = setup_vs[vs]
    lotus.settings.configure(rm=rm, vs=my_vs)

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

@pytest.mark.parametrize("vs", VECTOR_STORE_TO_CLS.keys())
@pytest.mark.parametrize("model", get_enabled("intfloat/e5-small-v2", "text-embedding-3-small"))
def test_vs_search_rm_only(setup_models, setup_vs, vs, model):
    rm = setup_models[model]
    my_vs = setup_vs[vs]
    lotus.settings.configure(rm=rm, vs=my_vs)

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
    df = df.sem_search("Course Name", "Optimization", K=1)
    assert df["Course Name"].tolist() == ["Optimization Methods in Engineering"]

@pytest.mark.parametrize("vs", VECTOR_STORE_TO_CLS.keys())
@pytest.mark.parametrize("model", get_enabled("intfloat/e5-small-v2", "text-embedding-3-small"))
def test_vs_sim_join(setup_models, setup_vs, vs, model):
    rm = setup_models[model]
    my_vs = setup_vs[vs]
    lotus.settings.configure(rm=rm, vs=my_vs)

    data1 = {
        "Course Name": [
            "History of the Atlantic World",
            "Riemannian Geometry",
        ]
    }

    data2 = {"Skill": ["Math", "History"]}

    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2).sem_index("Skill", "indexdir")
    joined_df = df1.sem_sim_join(df2, left_on="Course Name", right_on="Skill", K=1)
    joined_pairs = set(zip(joined_df["Course Name"], joined_df["Skill"]))
    expected_pairs = {("History of the Atlantic World", "History"), ("Riemannian Geometry", "Math")}
    assert joined_pairs == expected_pairs, joined_pairs


# TODO: threshold is hardcoded for intfloat/e5-small-v2
@pytest.mark.skipif(
    "intfloat/e5-small-v2" not in ENABLED_MODEL_NAMES,
    reason="Skipping test because intfloat/e5-small-v2 is not enabled",
)
@pytest.mark.parametrize("vs", VECTOR_STORE_TO_CLS.keys())
def test_vs_dedup(setup_models, setup_vs, vs):
    rm = setup_models["intfloat/e5-small-v2"]
    my_vs = setup_vs[vs]
    lotus.settings.configure(rm=rm, vs=my_vs)
    data = {
        "Text": [
            "Probability and Random Processes",
            "Probability and Markov Chains",
            "Harry Potter",
            "Harry James Potter",
        ]
    }
    df = pd.DataFrame(data)
    df = df.sem_index("Text", "indexdir").sem_dedup("Text", threshold=0.85)
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
    lotus.settings.configure(rm=rm, vs = vs, reranker=reranker)

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
