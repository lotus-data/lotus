"""Tests for deepseek model support across semantic operators."""

import pandas as pd
import pytest

import lotus
from lotus.models import LM

def test_model_config():
    """Test that deepseek models are configured correctly"""
    # Deepseek model should set temperature=0.6 and strategy=deepseek
    lm = LM(model="ollama/deepseek-r1:7b")
    assert lm.kwargs.get("strategy") == "deepseek"
    assert lm.kwargs.get("temperature") == 0.6  # Recommended temperature
    assert lm.is_deepseek  # Should detect deepseek model

    # Non-deepseek model should keep original settings
    lm = LM(model="ollama/llama3.1:8b", temperature=0.7)
    assert "strategy" not in lm.kwargs
    assert lm.kwargs.get("temperature") == 0.7
    assert not lm.is_deepseek  # Should not detect as deepseek

def test_map_behavior():
    """Test that sem_map handles both deepseek and non-deepseek models correctly"""
    df = pd.DataFrame({
        "Course": ["Machine Learning", "Data Structures"]
    })

    # Test deepseek model
    lm = LM(model="ollama/deepseek-r1:7b")
    lotus.settings.configure(lm=lm)
    
    result = df.sem_map(
        "What is a similar course to {Course}?",
        return_explanations=True,
        strategy="deepseek"
    )
    
    # Should have reasoning in explanation_map
    assert "explanation_map" in result.columns
    assert result["explanation_map"].iloc[0] is not None
    assert isinstance(result["_map"].iloc[0], str)

    # Test non-deepseek model
    lm = LM(model="ollama/llama3.1:8b")
    lotus.settings.configure(lm=lm)
    
    result = df.sem_map(
        "What is a similar course to {Course}?",
        return_explanations=True  # Non-deepseek model should not use deepseek strategy
    )
    
    # Should have no reasoning
    assert result["explanation_map"].iloc[0] is None
    assert isinstance(result["_map"].iloc[0], str)

def test_filter_behavior():
    """Test that sem_filter handles both deepseek and non-deepseek models correctly"""
    df = pd.DataFrame({
        "Course": ["Machine Learning", "Art History"]
    })

    # Test deepseek model
    lm = LM(model="ollama/deepseek-r1:7b")
    lotus.settings.configure(lm=lm)
    
    result = df.sem_filter(
        "Is {Course} a technical course?",
        return_explanations=True,
        strategy="deepseek"
    )
    
    # Should have reasoning in explanation_filter
    assert "explanation_filter" in result.columns
    filtered_rows = result[result["Course"] == "Machine Learning"]
    assert len(filtered_rows) > 0
    assert filtered_rows["explanation_filter"].iloc[0] is not None

    # Test non-deepseek model
    lm = LM(model="ollama/llama3.1:8b")
    lotus.settings.configure(lm=lm)
    
    result = df.sem_filter(
        "Is {Course} a technical course?",
        return_explanations=True  # Non-deepseek model should not use deepseek strategy
    )
    
    # Should have no reasoning
    filtered_rows = result[result["Course"] == "Machine Learning"]
    assert len(filtered_rows) > 0
    assert filtered_rows["explanation_filter"].iloc[0] is None

def test_join_behavior():
    """Test that sem_join handles both deepseek and non-deepseek models correctly"""
    df1 = pd.DataFrame({
        "Course1": ["Machine Learning"]
    })
    df2 = pd.DataFrame({
        "Course2": ["Statistics", "Art History"]
    })

    # Test deepseek model
    lm = LM(model="ollama/deepseek-r1:7b")
    lotus.settings.configure(lm=lm)
    
    result = df1.sem_join(
        df2,
        "{Course1} and {Course2} are related fields",
        return_explanations=True,
        strategy="deepseek"
    )
    
    # Should have reasoning in explanation_join
    assert "explanation_join" in result.columns
    assert len(result) > 0
    assert result["explanation_join"].iloc[0] is not None

    # Test non-deepseek model
    lm = LM(model="ollama/llama3.1:8b")
    lotus.settings.configure(lm=lm)
    
    result = df1.sem_join(
        df2,
        "{Course1} and {Course2} are related fields",
        return_explanations=True  # Non-deepseek model should not use deepseek strategy
    )
    
    # Should have no reasoning
    assert len(result) > 0
    assert result["explanation_join"].iloc[0] is None

def test_extract_behavior():
    """Test that sem_extract handles both deepseek and non-deepseek models correctly"""
    df = pd.DataFrame({
        "Text": ["The course Machine Learning (CS229) is taught by Prof. Smith"]
    })

    # Test deepseek model
    lm = LM(model="ollama/deepseek-r1:7b")
    lotus.settings.configure(lm=lm)
    
    result = df.sem_extract(
        input_cols=["Text"],
        output_cols={
            "course_code": "Course code",
            "professor": "Professor name"
        },
        return_raw_outputs=True,
        strategy="deepseek"
    )
    
    # Should extract fields correctly with reasoning
    assert "course_code" in result.columns
    assert "professor" in result.columns
    assert isinstance(result["course_code"].iloc[0], str)
    assert isinstance(result["professor"].iloc[0], str)
    assert "<think>" in result["raw_output"].iloc[0]

    # Test non-deepseek model
    lm = LM(model="ollama/llama3.1:8b")
    lotus.settings.configure(lm=lm)
    
    result = df.sem_extract(
        input_cols=["Text"],
        output_cols={
            "course_code": "Course code",
            "professor": "Professor name"
        },
        return_raw_outputs=True
    )
    
    # Should extract fields correctly without reasoning
    assert "course_code" in result.columns
    assert "professor" in result.columns
    assert isinstance(result["course_code"].iloc[0], str)
    assert isinstance(result["professor"].iloc[0], str)
    assert "<think>" not in result["raw_output"].iloc[0]