import os
import sqlite3

import boto3
import pandas as pd
import pytest

import lotus
from lotus.data_connectors import DataConnector
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


@pytest.fixture(scope="session")
def setup_minio():
    minio_config = {
        "aws_access_key": "accesskey",
        "aws_secret_key": "secretkey",
        "region": None,
        "bucket": "test-bucket",
        "file_path": "data/test.csv",
        "protocol": "http",
        "endpoint_url": "http://localhost:9000",
    }

    session = boto3.Session(
        aws_access_key_id=minio_config["aws_access_key"],
        aws_secret_access_key=minio_config["aws_secret_key"],
    )

    s3 = session.resource("s3", endpoint_url=minio_config["endpoint_url"])

    try:
        s3.create_bucket(Bucket=minio_config["bucket"])
    except s3.meta.client.exceptions.BucketAlreadyOwnedByYou:
        pass

    # Upload test file
    test_data = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "score": [85, 90, 88],
        }
    )
    csv_data = test_data.to_csv(index=False)

    s3.Bucket(minio_config["bucket"]).put_object(Key="test_data.csv", Body=csv_data)

    return minio_config


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

    query = "SELECT * FROM movies"
    df = DataConnector.load_from_db("sqlite:///example_movies.db", query=query)
    assert len(df) > 0

    filtered_df = df.sem_filter("{name} is an adult")
    assert isinstance(filtered_df, pd.DataFrame)


@pytest.mark.parametrize("model", get_enabled("gpt-4o-mini"))
def test_minio(setup_models, setup_minio, model):
    lm = setup_models[model]
    lotus.settings.configure(lm=lm)
    minio_config = setup_minio

    df = DataConnector.load_from_s3(
        aws_access_key=minio_config["aws_access_key"],
        aws_secret_key=minio_config["aws_secret_key"],
        region=minio_config["region"],
        bucket=minio_config["bucket"],
        file_path="test_data.csv",
        endpoint_url=minio_config["endpoint_url"],
        protocol="http",
    )

    assert not df.empty
    assert df.shape[0] == 3
    assert set(df.columns) == {"id", "name", "score"}
