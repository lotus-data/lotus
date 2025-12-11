import time
from typing import Type

import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

import lotus
from lotus.models import LM


class FileReadArgs(BaseModel):
    filename: str = Field(..., description="The name of the file to read.")


class FileReadTool(BaseTool):
    name: str = "File Read"
    description: str = "A tool to read files from the local filesystem."
    args_schema: Type[BaseModel] = FileReadArgs

    def __init__(self):
        super().__init__()

    def _run(self, filename: str) -> str:
        print(f"Reading file: {filename}")
        time.sleep(5)
        if filename == "text1.txt":
            return "13 + 16"
        elif filename == "text2.txt":
            return "29 + 51"
        return "<File not found>"


class AdditionArgs(BaseModel):
    num1: str = Field(..., description="The first number.")
    num2: str = Field(..., description="The second number.")


class AdditionTool(BaseTool):
    name: str = "Addition Tool"
    description: str = "A tool to add two numbers."
    args_schema: Type[BaseModel] = AdditionArgs

    def __init__(self):
        super().__init__()

    def _run(self, num1: int, num2: int) -> str:
        print(f"Adding numbers: {num1} + {num2}")
        time.sleep(5)
        return str(int(num1) + int(num2))


lm = LM()

lotus.settings.configure(lm=lm)
data = {
    "File names": [
        "text1.txt",
        "text2.txt",
    ]
}

df = pd.DataFrame(data)
user_instruction = "Evaluate the mathametical expression in the file {File names} to a single integer. Give only the integer as output."
df = df.sem_map(user_instruction, tools=[FileReadTool(), AdditionTool()])
print(df)
