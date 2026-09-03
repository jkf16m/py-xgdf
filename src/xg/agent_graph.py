"""LangGraph building blocks for xg: graphs are the main unit.

All xg graphs agree on one state schema (:class:`GraphState`), so any graph
here can be composed with any other graph without adapters.

    GraphState:  messages, request, cfg (AgentConfig), session (Session)

Provided graphs:
    build_workspace_read_graph()  pre-loads every workspace file into the
                                  session as simulated read tool calls
    build_agent_graph(tools)      the agent loop: LLM → router → tools → ...
    build_default_graph(tools)    workspace read → prompt → agent loop

cfg exposes a writable session reference: ``cfg.session`` (created lazily
by ``cfg.get_session()``); nodes append to it directly.
"""

from __future__ import annotations

import json
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from xg.graphs import AgentConfig, Session
from xg.tools import ToolState, dispatch, schemas
from xg.utils import tool_patch, format_patch


class GraphState(TypedDict, total=False):
    """The one state schema every xg graph agrees on."""
    messages: list[dict]
    request: str
    cfg: AgentConfig
    session: Session


def _tool_state(cfg: AgentConfig) -> ToolState:
    """One editing state machine per cfg, kept on the runtime.

    The state machine carries the selected file across turns (select →
    edit), so it must outlive individual node invocations.
    """
    runtime = cfg._runtime
    state = getattr(runtime, "_tool_state", None)
    if state is None:
        state = ToolState(runtime.root)
        runtime._tool_state = state
    return state


# ---- nodes ---------------------------------------------------------------


def workspace_read_node(state: GraphState, config) -> dict:
    """Pre-load the whole workspace into the session as read tool calls.

    Each readable file becomes one assistant message carrying a single
    deterministic ``read`` tool call plus its tool result with the raw
    file content. Happens once per session window; a repeat is a no-op.
    """
    from xg.utils import read_workspace

    cfg: AgentConfig = state["cfg"]
    session = cfg.get_session()
    read_workspace(session, cfg.root)
    return {"session": session}


def user_prompt_node(state: GraphState, config) -> dict:
    """Resolve the request: explicit, resumed, or asked from the user."""
    cfg: AgentConfig = state["cfg"]
    session = cfg.get_session()

    request = state.get("request") or cfg.request
    if not request:
        if not cfg.prompt("your request", session=session):
            return {"request": ""}
        request = cfg.request

    if session.path is not None and not state.get("request"):
        session.add("user", request)
        session.append({"xgdf-prompt": request})  # resume event

    return {"request": request, "messages": [{"role": "user", "content": request}]}


def llm_request_node(state: GraphState, config) -> dict:
    """One streaming chat request over the session; appends the reply."""
    from xg.llm import chat

    cfg: AgentConfig = state["cfg"]
    session = cfg.get_session()
    tool_state = _tool_state(cfg)
    reply = chat(session, schemas(tool_state.allowed_tools()))
    session.append(reply)
    return {"messages": [reply], "session": session}


def route_by_tool_calls(state: GraphState, config) -> str:
    """Loop while the last assistant message wants tools; stop when done.

    The session is the source of truth — no extra routing flags in state.
    """
    last = state["session"].last() or {}
    if last.get("tool_calls"):
        return "tool_executor"
    return "end"


def tool_executor_node(state: GraphState, config) -> dict:
    """Run the model's tool calls through the ToolState state machine.

    Each call is previewed as a git patch and gated on user confirmation;
    a rejected call ends the loop (its result tells the model so).
    """
    from xg.agent import ToolRejected

    cfg: AgentConfig = state["cfg"]
    session = cfg.get_session()
    state = _tool_state(cfg)
    last = session.last()
    tool_calls = last.get("tool_calls") or []

    for call in tool_calls:
        fn = call.get("function") or {}
        name = fn.get("name", "")
        print(f"\nProposed tool call: {name}({fn.get('arguments', '{}')})")
        patch = tool_patch(state, name, fn.get("arguments", "{}"))
        if patch:
            print(format_patch(patch))
        try:
            if not _accept_tool():
                raise ToolRejected(name)
            result = dispatch(state, name, fn.get("arguments", "{}"))
        except ToolRejected as exc:
            print(f"\n[graph] tool call rejected: {exc}")
            session.append({"role": "tool", "tool_call_id": call.get("id", ""),
                            "content": f"user rejected the tool call: {name}"})
            return {"messages": [], "session": session}
        except Exception as exc:
            result = f"tool error: {exc}"
        session.append({"role": "tool", "tool_call_id": call.get("id", ""),
                        "content": result})
        print(f"\n[graph] {name}: {result[:200]}")
    return {"messages": [], "session": session}


def _accept_tool() -> bool:
    """Terminal confirmation gate, mirroring the legacy loop."""
    import sys

    if not sys.stdin.isatty():
        return False  # non-interactive: reject
    try:
        answer = input("apply? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer in {"y", "yes"}


# ---- graphs --------------------------------------------------------------


def build_workspace_read_graph():
    """START → workspace_read → END. Composable pre-load step."""
    graph = StateGraph(GraphState)
    graph.add_node("workspace_read", workspace_read_node)
    graph.add_edge(START, "workspace_read")
    graph.add_edge("workspace_read", END)
    return graph.compile()


def build_agent_graph(tools=None):
    """The agent loop: LLM → router → tool_executor → LLM → ... → END.

    Without tools there is nothing to execute, so the loop is direct:
    LLM → END.
    """
    graph = StateGraph(GraphState)
    graph.add_node("llm_request", llm_request_node)
    if tools:
        graph.add_node("tool_executor", tool_executor_node)
        graph.add_conditional_edges(
            "llm_request", route_by_tool_calls,
            {"tool_executor": "tool_executor", "end": END},
        )
        graph.add_edge("tool_executor", "llm_request")
    else:
        graph.add_edge("llm_request", END)
    graph.add_edge(START, "llm_request")
    return graph.compile()


def build_default_graph(tools=None):
    """The default graph: workspace read → prompt → agent loop.

    Composed from the reusable graphs above — same shared state, no glue.
    """
    graph = StateGraph(GraphState)
    graph.add_node("workspace_read", build_workspace_read_graph())
    graph.add_node("user_prompt", user_prompt_node)
    graph.add_node("agent", build_agent_graph(tools))
    graph.add_edge(START, "workspace_read")
    graph.add_edge("workspace_read", "user_prompt")
    graph.add_edge("user_prompt", "agent")
    graph.add_edge("agent", END)
    return graph.compile()
