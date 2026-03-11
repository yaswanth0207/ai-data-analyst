"""
LangGraph agent graph: load_data → route → generate_code → execute → check_error → fix_code (retry) | generate_answer.

Note: we intentionally do NOT use a persistent checkpointer here, because
the state includes a pandas DataFrame which is not msgpack-serializable.
State lives only in memory per request.
"""
from langgraph.graph import END, StateGraph

from backend.agent.state import AgentState
from backend.agent import nodes


def _start_router(state: AgentState) -> str:
    """If dataframe and schema already in state (e.g. from session), skip load_data."""
    if state.get("dataframe") is not None and state.get("schema"):
        return "route_question"
    return "load_data"


def build_agent_graph():
    """Build and return the compiled LangGraph agent."""
    builder = StateGraph(AgentState)

    # Nodes
    builder.add_node("load_data", nodes.load_data)
    builder.add_node("route_question", nodes.route_question)
    builder.add_node("generate_code", nodes.generate_code_node)
    builder.add_node("execute_code", nodes.execute_code_node)
    builder.add_node("fix_code", nodes.fix_code)
    builder.add_node("generate_answer", nodes.generate_answer)

    # Flow: START → load_data or route_question (if state has data) → ...
    builder.set_conditional_entry_point(_start_router, ["load_data", "route_question"])
    builder.add_edge("load_data", "route_question")  # after load_data always route_question
    builder.add_edge("route_question", "generate_code")
    builder.add_edge("generate_code", "execute_code")
    builder.add_conditional_edges("execute_code", _route_after_execute, ["generate_answer", "fix_code"])
    builder.add_edge("fix_code", "execute_code")
    builder.add_edge("generate_answer", END)

    # No checkpointer: DataFrames in state are not msgpack-serializable.
    return builder.compile()


def _route_after_execute(state: AgentState) -> str:
    """After execute_code: if error and retries left, go to fix_code; else generate_answer."""
    err = state.get("error")
    retry = state.get("retry_count") or 0
    max_retries = int(__import__("os").getenv("MAX_RETRIES", "2"))
    if err and retry < max_retries:
        return "fix_code"
    return "generate_answer"


# Single global graph instance for the API
agent_graph = build_agent_graph()
