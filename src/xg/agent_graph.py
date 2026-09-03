"""LangGraph agent loop: LLM → router → tool_executor → LLM → ... → END

Simple agent loop that can be executed directly or composed with other graphs.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import StateGraph, START, END


class AgentState(TypedDict):
    messages: list[dict]
    cfg: dict
    session: str


def _llm_request(state: AgentState, config) -> dict:
    """Call LLM, append assistant response."""
    from xg.llm import request
    from xg.workflows import AgentConfig

    cfg = AgentConfig(**state["cfg"])
    messages = list(state["messages"])

    response = request(
        model=cfg.model,
        messages=messages,
        tools=None,
        stream=False,
    )

    messages.append({
        "role": "assistant",
        "content": response.content or "",
        "tool_calls": response.tool_calls or None,
        "stop_reason": "tool_calls" if response.tool_calls else "stop",
    })

    return {"messages": messages}


def _router(state: AgentState, config) -> str:
    """Route based on stop_reason."""
    messages = state["messages"]
    if not messages:
        return "end"
    last = messages[-1]
    if last.get("stop_reason") == "tool_calls":
        return "tool_executor"
    return "end"


def _tool_executor(state: AgentState, config) -> dict:
    """Execute tool calls from last assistant message."""
    from xg.sdk import ToolRegistry
    from xg.workflows import AgentConfig

    cfg = AgentConfig(**state["cfg"])
    messages = list(state["messages"])
    last = messages[-1]
    tool_calls = last.get("tool_calls", [])
    registry = ToolRegistry.all()

    for tc in tool_calls:
        fn_name = tc["function"]["name"]
        import json
        args = json.loads(tc["function"]["arguments"])
        tool = registry.get(fn_name)
        try:
            result = tool.run(args, cfg) if tool else f"error: unknown tool {fn_name!r}"
        except Exception as exc:
            result = f"error: {exc}"
        messages.append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": str(result),
        })

    return {"messages": messages}


def build_agent_graph():
    """Build and compile the agent loop graph."""
    graph = StateGraph(AgentState)
    graph.add_node("llm_request", _llm_request)
    graph.add_node("tool_executor", _tool_executor)
    graph.add_conditional_edges(
        "llm_request", _router,
        {"tool_executor": "tool_executor", "end": END},
    )
    graph.add_edge("tool_executor", "llm_request")
    graph.add_edge(START, "llm_request")
    return graph.compile()


AGENT_GRAPH = build_agent_graph()
