from typing import Type
from unittest.mock import MagicMock, patch

import pandas as pd
from crewai import CrewOutput
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

import lotus
from lotus.models import LM, LMWithTools
from tests.base_test import BaseTest


class FileReadArgs(BaseModel):
    filename: str = Field(..., description="The name of the file to read.")


class FileReadTool(BaseTool):
    name: str = "File Read"
    description: str = "A tool to read files from the local filesystem."
    args_schema: Type[BaseModel] = FileReadArgs

    def __init__(self):
        super().__init__()

    def _run(self, filename: str) -> str:
        if filename == "text1.txt":
            return "<Contents of text1>"
        elif filename == "text2.txt":
            return "<Contents of text2>"
        return "<File not found>"


class AsyncMock(MagicMock):
    async def __call__(self, *args, **kwargs):
        return super(AsyncMock, self).__call__(*args, **kwargs)


class TestFilters(BaseTest):
    def test_sem_map(self):
        """Test that the default LM with tools is None"""
        lotus.settings.configure(lm=LM(model="gpt-4o-mini"))
        assert lotus.settings.lm is not None
        df = pd.DataFrame(
            {
                "Course Name": [
                    "Probability and Random Processes",
                    "Optimization Methods in Engineering",
                ]
            }
        )

        user_instruction = "What is a similar course to {Course Name}. Be concise."
        df = df.sem_map(user_instruction, suffix="similar_course")
        assert "similar_course" in df.columns

    @patch("crewai.crew.Crew.kickoff_for_each_async", new_callable=AsyncMock)
    def test_sem_map_with_tools(self, mock_kickoff):
        """Test configuring the LM with tools in settings"""
        lotus.settings.configure(lm_with_tools=LMWithTools(model="gpt-4o-mini"))
        assert lotus.settings.lm_with_tools is not None
        file_read_tool = FileReadTool()
        df = pd.DataFrame(
            {
                "File names": [
                    "text1.txt",
                    "text2.txt",
                ]
            }
        )

        mock_kickoff.return_value = [CrewOutput(raw="<Contents of text1>"), CrewOutput(raw="<Contents of text2>")]

        df = df.sem_map(
            "Read the file {File names} and return its contents.",
            tools=[file_read_tool],
            suffix="file_contents",
        )
        assert "file_contents" in df.columns
