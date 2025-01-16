import os
import sqlite3

import pandas as pd
import pytest

import lotus
from lotus.databases import DatabaseConnector, LotusDB
from lotus.models import LM

################################################################################
# Setup
################################################################################
# Set logger level to DEBUG
lotus.logger.setLevel("DEBUG")

# Environment flags to enable/disable tests
ENABLE_OPENAI_TESTS = os.getenv("ENABLE_OPENAI_TESTS", "false").lower() == "true"
ENABLE_OLLAMA_TESTS = os.getenv("ENABLE_OLLAMA_TESTS", "false").lower() == "true"

MODEL_NAME_TO_ENABLED = {
    "gpt-4o-mini": ENABLE_OPENAI_TESTS,
    "gpt-4o": ENABLE_OPENAI_TESTS,
    "ollama/llama3.1": ENABLE_OLLAMA_TESTS,
}
ENABLED_MODEL_NAMES = set([model_name for model_name, is_enabled in MODEL_NAME_TO_ENABLED.items() if is_enabled])


def get_enabled(*candidate_models: str) -> list[str]:
    return [model for model in candidate_models if model in ENABLED_MODEL_NAMES]


@pytest.fixture(scope="session")
def setup_models():
    models = {}

    for model_path in ENABLED_MODEL_NAMES:
        models[model_path] = LM(model=model_path)

    return models


@pytest.fixture(scope="session")
def setup_sqlite_db():
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE test_table (
        id INTEGER PRIMARY KEY,
        name TEXT,
        age INTEGER
        )
    """)

    cursor.executemany(
        """INSERT INTO test_table (name, age) VALUES (?, ?)""",
        [("Alice", 8), ("Bob", 14), ("Charlie", 35), ("Dave", 42)],
    )

    conn.commit()
    conn.close()


@pytest.fixture(autouse=True)
def print_usage_after_each_test(setup_models):
    yield  # this runs the test
    models = setup_models
    for model_name, model in models.items():
        print(f"\nUsage stats for {model_name} after test:")
        model.print_total_usage()
        model.reset_stats()
        model.reset_cache()


#################################################################################
# Standard Tests
#################################################################################


@pytest.mark.parametrize("model", get_enabled("gpt-4o-mini"))
def test_SQL_db(setup_models, model):
    lm = setup_models[model]
    lotus.settings.configure(lm=lm)

    connector = DatabaseConnector()
    connector.connect_sql("sqlite:///:memory:")
    lotus_db = LotusDB(connector)

    df = lotus_db.query("SELECT * FROM test_table")
    assert len(df) > 0

    filtered_df = df.sem_filter("{name} is an adult")
    assert isinstance(filtered_df, pd.DataFrame)
