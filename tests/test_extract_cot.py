import os

import pandas as pd
import pytest

import lotus
from lotus.models import LM
from lotus.types import ReasoningStrategy

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


@pytest.fixture(autouse=True)
def print_usage_after_each_test(setup_models):
    yield  # this runs the test
    models = setup_models
    for model_name, model in models.items():
        print(f"\nUsage stats for {model_name} after test:")
        model.print_total_usage()
        model.reset_stats()
        model.reset_cache()


################################################################################
# Extract CoT tests
################################################################################
@pytest.mark.parametrize("model", get_enabled("gpt-4o-mini"))
def test_sem_extract_cot_basic(setup_models, model):
    """Test basic sem_extract with CoT reasoning."""
    lm = setup_models[model]
    lotus.settings.configure(lm=lm)

    data = {
        "Text": [
            "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity",
            "Marie Curie was a Polish-French physicist and chemist who conducted pioneering research on radioactivity",
            "Isaac Newton was an English mathematician, physicist, and astronomer who formulated the laws of motion",
        ]
    }
    df = pd.DataFrame(data)
    input_cols = ["Text"]
    output_cols = {
        "Name": "The person's full name",
        "Profession": "The person's primary profession",
        "Nationality": "The person's nationality or origin",
    }

    # Test with CoT reasoning and explanations
    result_df = df.sem_extract(
        input_cols, output_cols, strategy=ReasoningStrategy.ZS_COT, return_explanations=True, extract_quotes=False
    )

    # Verify structure
    assert len(result_df) == 3
    assert "Name" in result_df.columns
    assert "Profession" in result_df.columns
    assert "Nationality" in result_df.columns
    assert "explanation" in result_df.columns

    # Verify content (case-insensitive matching)
    expected_names = ["albert einstein", "marie curie", "isaac newton"]
    actual_names = [str(name).lower().strip() for name in result_df["Name"].tolist()]
    for expected, actual in zip(expected_names, actual_names):
        assert expected in actual, f"Expected '{expected}' to be in '{actual}'"

    # Verify explanations are present (may be empty for some models)
    for explanation in result_df["explanation"]:
        assert explanation is not None, "Explanation should not be None"
        # Note: explanations may be empty strings for some models/configurations


@pytest.mark.parametrize("model", get_enabled("gpt-4o-mini"))
def test_sem_extract_cot_with_quotes(setup_models, model):
    """Test sem_extract with CoT reasoning and quote extraction."""
    lm = setup_models[model]
    lotus.settings.configure(lm=lm)

    data = {
        "Movie": [
            "The Dark Knight is a 2008 superhero film directed by Christopher Nolan starring Christian Bale",
            "Pulp Fiction is a 1994 crime film written and directed by Quentin Tarantino featuring John Travolta",
        ]
    }
    df = pd.DataFrame(data)
    input_cols = ["Movie"]
    output_cols = {
        "director": "The director of the movie",
        "year": "The release year",
    }

    # Test with CoT reasoning, explanations, and quotes
    result_df = df.sem_extract(
        input_cols, output_cols, strategy=ReasoningStrategy.ZS_COT, return_explanations=True, extract_quotes=True
    )

    # Verify structure
    assert len(result_df) == 2
    assert "director" in result_df.columns
    assert "year" in result_df.columns
    assert "director_quote" in result_df.columns
    assert "year_quote" in result_df.columns
    assert "explanation" in result_df.columns

    # Verify directors
    expected_directors = ["christopher nolan", "quentin tarantino"]
    actual_directors = [str(director).lower().strip() for director in result_df["director"].tolist()]
    for expected, actual in zip(expected_directors, actual_directors):
        assert expected in actual, f"Expected '{expected}' to be in '{actual}'"

    # Verify years
    expected_years = ["2008", "1994"]
    actual_years = [str(year).strip() for year in result_df["year"].tolist()]
    for expected, actual in zip(expected_years, actual_years):
        assert expected in actual, f"Expected '{expected}' to be in '{actual}'"

    # Verify quotes contain the extracted values
    for idx, row in result_df.iterrows():
        director_name = str(row["director"]).lower()
        director_quote = str(row["director_quote"]).lower()
        assert director_name in director_quote, f"Director '{director_name}' not found in quote '{director_quote}'"

        year = str(row["year"])
        year_quote = str(row["year_quote"])
        assert year in year_quote, f"Year '{year}' not found in quote '{year_quote}'"

        # Verify explanations are present (may be empty for some models)
        for explanation in result_df["explanation"]:
            assert explanation is not None, "Explanation should not be None"
            # Note: explanations may be empty strings for some models/configurations


@pytest.mark.parametrize("model", get_enabled("gpt-4o-mini"))
def test_sem_extract_cot_consistency(setup_models, model):
    """Test that CoT extract produces consistent output formats."""
    lm = setup_models[model]
    lotus.settings.configure(lm=lm)

    data = {
        "Description": [
            "John Smith is 25 years old and works as a software engineer",
            "Sarah Johnson is 30 years old and works as a doctor",
            "Mike Brown is 28 years old and works as a teacher",
        ]
    }
    df = pd.DataFrame(data)
    input_cols = ["Description"]
    output_cols = {
        "name": "The person's name",
        "age": "The person's age",
        "job": "The person's job",
    }

    # Test with CoT reasoning
    result_df = df.sem_extract(
        input_cols, output_cols, strategy=ReasoningStrategy.ZS_COT, return_explanations=True, extract_quotes=False
    )

    # Verify all outputs are consistent (strings)
    for col in output_cols.keys():
        for value in result_df[col]:
            assert isinstance(value, str), f"Value '{value}' in column '{col}' should be a string, got {type(value)}"

    # Verify no unexpected columns with dictionary values
    for col in result_df.columns:
        if col not in ["Description", "explanation"]:  # Skip input and explanation columns
            for value in result_df[col]:
                assert not isinstance(value, dict), f"Found unexpected dict in column '{col}': {value}"


@pytest.mark.parametrize("model", get_enabled("gpt-4o-mini"))
def test_sem_extract_cot_vs_regular(setup_models, model):
    """Test that CoT and regular extract produce similar results but CoT includes explanations."""
    lm = setup_models[model]
    lotus.settings.configure(lm=lm)

    data = {
        "Text": [
            "Apple Inc. was founded by Steve Jobs in 1976 and is headquartered in Cupertino, California",
            "Microsoft was founded by Bill Gates in 1975 and is headquartered in Redmond, Washington",
        ]
    }
    df = pd.DataFrame(data)
    input_cols = ["Text"]
    output_cols = {
        "company": "The company name",
        "founder": "The founder's name",
    }

    # Regular extract
    regular_df = df.sem_extract(input_cols, output_cols, extract_quotes=False)

    # CoT extract
    cot_df = df.sem_extract(
        input_cols, output_cols, strategy=ReasoningStrategy.ZS_COT, return_explanations=True, extract_quotes=False
    )

    # Both should have same number of rows and basic columns
    assert len(regular_df) == len(cot_df)
    assert "company" in regular_df.columns and "company" in cot_df.columns
    assert "founder" in regular_df.columns and "founder" in cot_df.columns

    # CoT should have explanations, regular should not
    assert "explanation" not in regular_df.columns
    assert "explanation" in cot_df.columns

    # Both should extract similar information (allowing for minor variations)
    for i in range(len(df)):
        regular_company = str(regular_df.loc[i, "company"]).lower()
        cot_company = str(cot_df.loc[i, "company"]).lower()

        # Should contain similar company names
        if "apple" in regular_company or "apple" in cot_company:
            assert "apple" in regular_company and "apple" in cot_company
        if "microsoft" in regular_company or "microsoft" in cot_company:
            assert "microsoft" in regular_company and "microsoft" in cot_company


@pytest.mark.parametrize("model", get_enabled("gpt-4o-mini"))
def test_sem_extract_cot_empty_data(setup_models, model):
    """Test CoT extract with edge cases like empty data."""
    lm = setup_models[model]
    lotus.settings.configure(lm=lm)

    # Empty DataFrame
    empty_df = pd.DataFrame({"Text": []})
    result_df = empty_df.sem_extract(
        input_cols=["Text"],
        output_cols={"info": "Any information"},
        strategy=ReasoningStrategy.ZS_COT,
        return_explanations=True,
    )

    # With empty data, we should get back the original DataFrame structure
    assert len(result_df) == 0
    assert "Text" in result_df.columns


@pytest.mark.parametrize("model", get_enabled("gpt-4o-mini"))
def test_sem_extract_cot_missing_info(setup_models, model):
    """Test CoT extract when information is not clearly available."""
    lm = setup_models[model]
    lotus.settings.configure(lm=lm)

    data = {
        "Text": [
            "This is a random sentence with no specific information",
            "Another sentence that doesn't contain the requested data",
        ]
    }
    df = pd.DataFrame(data)
    input_cols = ["Text"]
    output_cols = {
        "person_name": "The name of a person mentioned",
        "date": "Any date mentioned",
    }

    # Test with CoT reasoning
    result_df = df.sem_extract(
        input_cols, output_cols, strategy=ReasoningStrategy.ZS_COT, return_explanations=True, extract_quotes=False
    )

    # Should still return results (even if empty/null values)
    assert len(result_df) == 2
    assert "person_name" in result_df.columns
    assert "date" in result_df.columns
    assert "explanation" in result_df.columns

    # All values should be strings (even if empty or "None")
    for col in ["person_name", "date"]:
        for value in result_df[col]:
            assert isinstance(value, str), f"Value should be string, got {type(value)}: {value}"
