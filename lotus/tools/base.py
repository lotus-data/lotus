"""LOTUS-native tool interface for agentic operators.

A ``Tool`` is a small pydantic-described callable that an agent can invoke during
the agentic map loop. Tools serialize to the OpenAI/litellm function-tool schema so
they work with native tool-calling. Two ways to define one:

    # (a) decorator — args schema inferred from the signature + type hints
    @tool(description="Add two integers and return the sum.")
    def add(a: int, b: int) -> str:
        return str(a + b)

    # (b) subclass — explicit schema for richer control
    class FileReadTool(Tool):
        name = "file_read"
        description = "Read a file from the sandbox filesystem."
        args_schema = FileReadArgs
        def run(self, filename: str) -> str:
            ...
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

from pydantic import BaseModel, create_model


class Tool:
    """Base class for agent tools.

    Subclasses set ``name``, ``description`` and ``args_schema`` (a pydantic model)
    and implement ``run(**kwargs) -> str``.
    """

    name: str = ""
    description: str = ""
    args_schema: type[BaseModel] | None = None

    def run(self, **kwargs: Any) -> str:  # pragma: no cover - overridden
        raise NotImplementedError("Tool subclasses must implement run().")

    def to_openai_schema(self) -> dict[str, Any]:
        """Serialize to the OpenAI/litellm function-tool schema."""
        if not self.name:
            raise ValueError("Tool.name must be set.")
        if self.args_schema is not None:
            parameters = self.args_schema.model_json_schema()
        else:
            parameters = {"type": "object", "properties": {}}
        # OpenAI does not want pydantic's extra "title" keys, but tolerates them.
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }

    def __repr__(self) -> str:
        return f"Tool(name={self.name!r})"


class _FunctionTool(Tool):
    """A Tool backed by a plain python function (created by ``@tool``)."""

    def __init__(self, fn: Callable[..., Any], name: str, description: str, args_schema: type[BaseModel]):
        self._fn = fn
        self.name = name
        self.description = description
        self.args_schema = args_schema

    def run(self, **kwargs: Any) -> str:
        result = self._fn(**kwargs)
        return result if isinstance(result, str) else str(result)


def _args_schema_from_signature(fn: Callable[..., Any], model_name: str) -> type[BaseModel]:
    """Build a pydantic args model from a function's signature + type hints."""
    sig = inspect.signature(fn)
    fields: dict[str, Any] = {}
    for pname, param in sig.parameters.items():
        if pname == "self" or param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        annotation = param.annotation if param.annotation is not inspect.Parameter.empty else str
        default = param.default if param.default is not inspect.Parameter.empty else ...
        fields[pname] = (annotation, default)
    return create_model(model_name, **fields)  # type: ignore[call-overload]


def tool(fn: Callable[..., Any] | None = None, *, name: str | None = None, description: str | None = None):
    """Decorator turning a function into a :class:`Tool`.

    The args schema is inferred from the signature + type hints; ``description``
    defaults to the function docstring.
    """

    def wrap(func: Callable[..., Any]) -> _FunctionTool:
        tool_name = name or func.__name__
        tool_desc = description or (inspect.getdoc(func) or "").strip()
        args_schema = _args_schema_from_signature(func, f"{tool_name}_Args")
        return _FunctionTool(func, tool_name, tool_desc, args_schema)

    if fn is not None:  # used as @tool without parentheses
        return wrap(fn)
    return wrap
