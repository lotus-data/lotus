"""Scripted-model tests through Corpus.agent and the real MCP SDK/HTTP transport."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock

import anyio
import httpx
import pytest

from lotus import Corpus
from lotus.agentic import Plan, run_agent
from lotus.agentic.loop import AgentStep, ToolCall
from lotus.tools import ParallelFetchTool, ParallelSearchTool, tool
from lotus.tools.base import _tool_session_id
from lotus.tools.parallel import _ENDPOINT, _USER_AGENT


@pytest.fixture
def mcp_http(monkeypatch):
    requests = []
    calls = []
    state = {"error": None, "delay": 0, "barrier": None}
    lock = Lock()
    original = httpx.AsyncClient

    async def handle(request):
        assert str(request.url) == _ENDPOINT
        assert request.headers["user-agent"] == _USER_AGENT
        assert "authorization" not in request.headers
        with lock:
            requests.append(request)
        if request.method != "POST":
            return httpx.Response(405)
        payload = json.loads(request.content)
        if "id" not in payload:
            return httpx.Response(202)
        method = payload["method"]
        if method == "initialize":
            result = {
                "protocolVersion": "2025-03-26",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "test", "version": "1"},
            }
        elif method == "tools/list":
            result = {
                "tools": [
                    {"name": t.name, "inputSchema": t.to_openai_schema()["function"]["parameters"]}
                    for t in [ParallelSearchTool(), ParallelFetchTool()]
                ]
            }
        elif method == "tools/call":
            params = payload["params"]
            with lock:
                calls.append(params)
            if state["barrier"] and params["name"] == "web_search":
                state["barrier"].wait(timeout=5)
            await anyio.sleep(state["delay"])
            if state.get("http_status"):
                return httpx.Response(state["http_status"], text="provider unavailable")
            error = state["error"]
            state["error"] = None
            result = {
                "content": [
                    {"type": "text", "text": error or "Source: https://example.org/evidence | LOTUS evidence excerpt"}
                ],
                "isError": bool(error),
            }
            if state.get("empty"):
                result["content"] = []
        else:
            raise AssertionError(method)
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": payload["id"], "result": result})

    class ObservedClient(original):
        def __init__(self, *args, **kwargs):
            assert kwargs["trust_env"] is True
            super().__init__(*args, **kwargs, transport=httpx.MockTransport(handle))

    monkeypatch.setattr(httpx, "AsyncClient", ObservedClient)
    # Ambient provider keys must never turn the anonymous default into auth.
    monkeypatch.setenv("PARALLEL_API_KEY", "ambient-key-must-not-be-used")
    return requests, calls, state


def factory(tools):
    assert [t.name for t in tools] == ["web_search", "web_fetch"]

    def complete(messages, *, tools_enabled=True):
        observations = [m["content"] for m in messages if m["role"] == "tool"]
        n = len(observations)
        if n in (0, 2):
            name, arguments = (
                "web_search",
                {"objective": "Find current LOTUS evidence", "search_queries": ["LOTUS research evidence"]},
            )
        elif n == 1:
            assert "https://example.org/evidence" in observations[0]
            name, arguments = "web_fetch", {"urls": ["https://example.org/evidence"]}
        else:
            assert all("LOTUS evidence excerpt" in x for x in observations)
            return AgentStep(content=observations[-1])
        return AgentStep(tool_calls=[ToolCall(str(n), name, arguments)])

    return complete


def research(**kwargs):
    return Corpus.from_documents(["LOTUS"]).agent(
        "Find supporting web evidence",
        ops=["map"],
        plan=Plan(ops=["map"], instructions={"map": "Research"}),
        lm=object(),
        completer_factory=factory,
        **kwargs,
    )


def test_default_caller_search_fetch_followup_and_reset(mcp_http):
    requests, calls, _ = mcp_http
    first = research()
    assert "https://example.org/evidence" in first.findings[0]
    assert "LOTUS evidence excerpt" in first.corpus.units[0].content
    assert [c["name"] for c in calls] == ["web_search", "web_fetch", "web_search"]
    assert len({c["arguments"]["session_id"] for c in calls}) == 1
    first_id = calls[0]["arguments"]["session_id"]
    research()
    assert len({c["arguments"]["session_id"] for c in calls[3:]}) == 1
    assert calls[3]["arguments"]["session_id"] != first_id
    assert _tool_session_id.get() is None
    assert any(json.loads(r.content).get("method") == "initialize" for r in requests if r.method == "POST")


def test_shared_tools_concurrent_agents_isolate_identity(mcp_http):
    _, calls, state = mcp_http
    state["barrier"] = Barrier(2)
    shared = [ParallelSearchTool(), ParallelFetchTool()]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: research(tools=shared), range(2)))
    assert all("LOTUS evidence excerpt" in r.findings[0] for r in results)
    ids = {c["arguments"]["session_id"] for c in calls}
    assert len(ids) == 2
    for session_id in ids:
        assert [c["name"] for c in calls if c["arguments"]["session_id"] == session_id] == [
            "web_search",
            "web_fetch",
            "web_search",
        ]


@pytest.mark.parametrize("selected", [[], [ParallelSearchTool()]])
def test_explicit_allowlist_preserved(selected, mcp_http):
    requests, _, _ = mcp_http

    def selected_factory(tools):
        assert tools is selected
        return lambda messages, **kwargs: AgentStep(content="No web access requested")

    result = Corpus.from_documents(["private data"]).agent(
        "Read only",
        ops=["map"],
        tools=selected,
        lm=object(),
        completer_factory=selected_factory,
        plan=Plan(ops=["map"], instructions={"map": "Read"}),
    )
    assert result.findings == ["No web access requested"]
    assert requests == []


def test_explicit_provider_and_denied_tool_are_not_replaced(mcp_http):
    requests, _, _ = mcp_http

    @tool(name="web_search", description="User-selected search")
    def saved_search(objective: str):
        return "saved provider"

    def complete(messages, **kwargs):
        if messages[-1]["role"] == "tool":
            return AgentStep(content=messages[-1]["content"])
        return AgentStep(tool_calls=[ToolCall("s", "web_search", {"objective": "query"})])

    assert run_agent(complete, [saved_search], "sys", "user").output == "saved provider"
    assert "unknown tool" in run_agent(complete, [], "sys", "user").output
    assert requests == []


def test_quota_error_and_retry_keep_session(mcp_http):
    _, calls, state = mcp_http
    state["error"] = "Anonymous free-tier quota exceeded"

    def complete(messages, **kwargs):
        observations = [m for m in messages if m["role"] == "tool"]
        if len(observations) == 2:
            assert "ERROR" in observations[0]["content"]
            assert "quota exceeded" in observations[0]["content"]
            return AgentStep(content=observations[1]["content"])
        return AgentStep(
            tool_calls=[
                ToolCall(
                    str(len(observations)),
                    "web_search",
                    {"objective": "Research", "search_queries": ["LOTUS evidence"]},
                )
            ]
        )

    assert "LOTUS evidence excerpt" in run_agent(complete, [ParallelSearchTool()], "sys", "user").output
    assert calls[0]["arguments"]["session_id"] == calls[1]["arguments"]["session_id"]


def test_timeout_is_consumed_by_agent(mcp_http):
    _, _, state = mcp_http
    state["delay"] = 0.3
    steps = iter(
        [
            AgentStep(tool_calls=[ToolCall("s", "web_search", {"objective": "x", "search_queries": ["x"]})]),
            AgentStep(content="Timed out"),
        ]
    )
    result = run_agent(lambda *a, **k: next(steps), [ParallelSearchTool(timeout=0.1)], "sys", "user")
    assert "TimeoutError" in result.trace[0]["result"]
    assert result.output == "Timed out"


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.org/file", "javascript:alert(1)"])
def test_invalid_urls_fail_before_network(url, mcp_http):
    requests, _, _ = mcp_http
    steps = iter([AgentStep(tool_calls=[ToolCall("f", "web_fetch", {"urls": [url]})]), AgentStep(content="invalid")])
    result = run_agent(lambda *a, **k: next(steps), [ParallelFetchTool()], "sys", "user")
    assert "ValidationError" in result.trace[0]["result"]
    assert requests == []


def test_cancellation_propagates_and_resets_context(monkeypatch):
    search = ParallelSearchTool()

    def cancel(**kwargs):
        assert _tool_session_id.get() is not None
        raise KeyboardInterrupt

    monkeypatch.setattr(search, "run", cancel)
    with pytest.raises(KeyboardInterrupt):
        run_agent(lambda *a, **k: AgentStep(tool_calls=[ToolCall("s", "web_search", {})]), [search], "sys", "user")
    assert _tool_session_id.get() is None


def test_portal_supports_caller_with_running_event_loop(mcp_http):
    async def caller():
        assert "LOTUS evidence excerpt" in research().findings[0]

    anyio.run(caller)


@pytest.mark.parametrize("failure", ["http", "empty"])
def test_transport_and_empty_result_errors_reach_agent(failure, mcp_http):
    _, _, state = mcp_http
    if failure == "http":
        state["http_status"] = 401
    else:
        state["empty"] = True
    steps = iter(
        [
            AgentStep(tool_calls=[ToolCall("s", "web_search", {"objective": "x", "search_queries": ["x"]})]),
            AgentStep(content="Search failed"),
        ]
    )
    result = run_agent(lambda *a, **k: next(steps), [ParallelSearchTool()], "sys", "user")
    assert "401" in result.trace[0]["result"] if failure == "http" else "no text" in result.trace[0]["result"]
    assert result.output == "Search failed"


def test_transport_respects_cancellation(mcp_http):
    _, _, state = mcp_http
    state["delay"] = 10

    async def cancel_request():
        with anyio.move_on_after(0.1) as scope:
            await ParallelSearchTool()._call({"objective": "x", "search_queries": ["x"], "session_id": "test-cancel"})
            pytest.fail("Cancelled transport returned a result")
        assert scope.cancel_called

    anyio.run(cancel_request)
