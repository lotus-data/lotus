"""Free anonymous web tools backed by Parallel's Search MCP server."""

from __future__ import annotations

from datetime import timedelta
from typing import Any
from uuid import uuid4

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field

from .base import Tool, _tool_session_id

_ENDPOINT = "https://search.parallel.ai/mcp"
_USER_AGENT = "lotus-ai (+https://github.com/lotus-data/lotus)"


def _error_text(error: BaseException) -> str:
    children = getattr(error, "exceptions", None)
    if children:
        return "; ".join(_error_text(child) for child in children)
    return f"{type(error).__name__}: {error}"


class SearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    objective: str = Field(min_length=1, description="The information to find.")
    search_queries: list[str] = Field(min_length=1, description="Related keyword queries of 3–6 words each.")


class FetchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    urls: list[AnyHttpUrl] = Field(min_length=1, max_length=20, description="HTTP(S) source URLs to read.")
    objective: str | None = Field(default=None, max_length=200)
    search_queries: list[str] | None = None


class _ParallelTool(Tool):
    remote_name: str

    def __init__(self, timeout: float = 30.0):
        if not 0 < timeout < float("inf"):
            raise ValueError("timeout must be positive and finite")
        self.timeout = timeout

    def run(self, **kwargs: Any) -> str:
        from anyio.from_thread import start_blocking_portal

        assert self.args_schema is not None
        arguments = self.args_schema.model_validate(kwargs).model_dump(mode="json", exclude_none=True)
        # The agent loop owns identity, independently of tools, connections and threads.
        # A direct standalone invocation is a separate research run.
        arguments["session_id"] = _tool_session_id.get() or str(uuid4())
        # A portal also supports synchronous Corpus.agent calls inside notebooks with
        # an already-running event loop. Context is passed explicitly in arguments.
        try:
            with start_blocking_portal() as portal:
                return portal.call(self._call, arguments)
        except Exception as error:
            # Async transport task groups may wrap the useful server/timeout error.
            raise RuntimeError(_error_text(error)) from error

    async def _call(self, arguments: dict[str, Any]) -> str:
        import anyio
        import httpx
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        with anyio.fail_after(self.timeout):
            async with httpx.AsyncClient(
                headers={"User-Agent": _USER_AGENT}, timeout=self.timeout, trust_env=True
            ) as http_client:
                async with streamable_http_client(_ENDPOINT, http_client=http_client) as (read, write, _):
                    async with ClientSession(
                        read, write, read_timeout_seconds=timedelta(seconds=self.timeout)
                    ) as session:
                        await session.initialize()
                        result = await session.call_tool(self.remote_name, arguments)
                        text = "\n".join(block.text for block in result.content if block.type == "text")
                        if result.isError:
                            raise RuntimeError(f"Parallel {self.remote_name}: {text}")
                        if not text:
                            raise RuntimeError(f"Parallel {self.remote_name} returned no text")
                        return text


class ParallelSearchTool(_ParallelTool):
    """Search the web anonymously using Parallel Fast mode. No API key needed."""

    name = "web_search"
    remote_name = "web_search"
    description = "Search the web for current evidence. Returns source URLs and excerpts."
    args_schema = SearchArgs


class ParallelFetchTool(_ParallelTool):
    """Read source pages through the anonymous Parallel Search MCP endpoint."""

    name = "web_fetch"
    remote_name = "web_fetch"
    description = "Fetch excerpts from HTTP(S) source pages, including URLs returned by web_search."
    args_schema = FetchArgs
