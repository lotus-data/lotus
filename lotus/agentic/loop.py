"""The per-item agentic tool-calling loop.

``run_agent`` drives one agent session: it sends messages to a ``Completer``, executes
any tool calls the model requests, feeds the results back, and repeats until the model
returns a final answer or ``max_steps`` is hit. This is the "map" worker in agentic
map-reduce, and also backs the low-level ``sem_map(tools=...)`` path.

The loop talks to the model through the ``Completer`` protocol so it is fully testable
without a network: production uses :class:`LiteLLMCompleter`; tests inject a fake that
returns scripted steps.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from lotus.tools.base import Tool


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class AgentStep:
    """One model turn: either final content, or a set of tool calls to execute."""

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)


class Completer(Protocol):
    """Sends the running message list to the model and returns the next step.

    Implementations are bound to a fixed tool-schema list at construction.
    """

    def __call__(self, messages: list[dict[str, Any]]) -> AgentStep: ...


@dataclass
class AgentResult:
    output: str
    trace: list[dict[str, Any]]
    steps: int
    truncated: bool
    usage: dict[str, int]


def run_agent(
    completer: Completer,
    tools: list[Tool],
    system_prompt: str,
    user_content: str,
    max_steps: int = 6,
) -> AgentResult:
    """Run a single agentic tool-calling session to completion."""
    tool_by_name = {t.name: t for t in tools}
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    trace: list[dict[str, Any]] = []
    usage: dict[str, int] = {}

    def _add_usage(u: dict[str, int]) -> None:
        for k, v in (u or {}).items():
            usage[k] = usage.get(k, 0) + v

    for step in range(max_steps):
        agent_step = completer(messages)
        _add_usage(agent_step.usage)

        if not agent_step.tool_calls:
            return AgentResult(agent_step.content or "", trace, step + 1, truncated=False, usage=usage)

        # Record the assistant's tool-call turn so the model sees its own request.
        messages.append(
            {
                "role": "assistant",
                "content": agent_step.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in agent_step.tool_calls
                ],
            }
        )

        for tc in agent_step.tool_calls:
            tool = tool_by_name.get(tc.name)
            if tool is None:
                result = f"ERROR: unknown tool '{tc.name}'"
            else:
                try:
                    result = tool.run(**tc.arguments)
                except Exception as e:  # tool failures are fed back, not fatal
                    result = f"ERROR: {type(e).__name__}: {e}"
            trace.append({"tool": tc.name, "arguments": tc.arguments, "result": result})
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": str(result)}
            )

    # Budget exhausted: one final tool-free turn to force an answer.
    final = completer(messages + [{"role": "user", "content": "Provide your final answer now."}])
    _add_usage(final.usage)
    return AgentResult(final.content or "", trace, max_steps, truncated=True, usage=usage)


class LiteLLMCompleter:
    """Production ``Completer`` backed by litellm native tool-calling.

    Reuses the configured LOTUS ``LM``'s model + kwargs (temperature, max tokens).
    """

    def __init__(self, lm: Any, tools: list[Tool] | None = None):
        self.lm = lm
        self.tool_schemas = [t.to_openai_schema() for t in (tools or [])] or None

    def __call__(self, messages: list[dict[str, Any]]) -> AgentStep:
        import litellm

        kwargs: dict[str, Any] = {}
        # Carry over the model's generation params (temperature, max_completion_tokens).
        for k in ("temperature", "max_completion_tokens"):
            if k in getattr(self.lm, "kwargs", {}):
                kwargs[k] = self.lm.kwargs[k]
        if self.tool_schemas:
            kwargs["tools"] = self.tool_schemas
            kwargs["tool_choice"] = "auto"

        resp = litellm.completion(model=self.lm.model, messages=messages, drop_params=True, **kwargs)
        choice = resp.choices[0]
        msg = choice.message
        tool_calls = []
        for tc in getattr(msg, "tool_calls", None) or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        usage = {}
        if getattr(resp, "usage", None) is not None:
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens or 0,
                "completion_tokens": resp.usage.completion_tokens or 0,
                "total_tokens": resp.usage.total_tokens or 0,
            }
        return AgentStep(content=msg.content, tool_calls=tool_calls, usage=usage)
