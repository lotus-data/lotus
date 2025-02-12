import pandas as pd
import pytest

import lotus
from lotus.dtype_extensions import DocumentArray, DocumentDtype
from lotus.settings import SerializationFormat
from lotus.templates.task_instructions import df2text
from tests.base_test import BaseTest


@pytest.fixture
def sample_df_with_pdfs():
    return pd.DataFrame(
        {
            "Title": ["Propositional Logic"],
            "Pdf": DocumentArray(["tests/assets/cs70-propositional-logic.pdf"]),
        }
    )


@pytest.fixture(autouse=True)
def reset_serialization_format():
    yield
    lotus.settings.serialization_format = SerializationFormat.DEFAULT


class TestSerialization(BaseTest):
    def test_df2text_default_format(self, sample_df):
        result = df2text(sample_df, ["Name", "Age"])
        expected = ["[Name]: «Alice»\n[Age]: «25»\n", "[Name]: «Bob»\n[Age]: «30»\n"]
        assert result == expected

    def test_df2text_json_format(self, sample_df):
        lotus.settings.serialization_format = SerializationFormat.JSON
        result = df2text(sample_df, ["Name", "Age"])
        expected = ['{"Name":"Alice","Age":25}', '{"Name":"Bob","Age":30}']
        assert result == expected

    def test_df2text_xml_format(self, sample_df):
        lotus.settings.serialization_format = SerializationFormat.XML
        result = df2text(sample_df, ["Name", "Age"])
        print(result)
        expected = [
            "<row><Name>Alice</Name><Age>25</Age></row>",
            "<row><Name>Bob</Name><Age>30</Age></row>",
        ]
        assert result == expected

    def test_df2text_nonexistent_columns(self, sample_df):
        result = df2text(sample_df, ["Name", "NonExistent"])
        expected = ["[Name]: «Alice»\n", "[Name]: «Bob»\n"]
        assert result == expected

    def test_df2text_empty_columns(self, sample_df):
        result = df2text(sample_df, [])
        assert result == ["", ""]

    def test_df2text_all_columns(self, sample_df):
        result = df2text(sample_df, ["Name", "Age", "City"])
        expected = [
            "[Name]: «Alice»\n[Age]: «25»\n[City]: «New York»\n",
            "[Name]: «Bob»\n[Age]: «30»\n[City]: «London»\n",
        ]
        assert result == expected

    def test_assert_documentdtype_recognition(self, sample_df_with_pdfs):
        print(sample_df_with_pdfs)
        assert isinstance(sample_df_with_pdfs["Pdf"].dtype, DocumentDtype)
