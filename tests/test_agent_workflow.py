import pandas as pd

from backend.agent.graph import build_agent_graph


def test_agent_workflow_runs_with_mocked_codegen(monkeypatch):
    # Avoid calling Ollama in tests by mocking code generation and answer generation.
    from backend.analyst import code_generator
    from backend.agent import nodes

    def fake_generate_code(user_question: str, schema: dict, mode: str) -> str:
        return "result = df['revenue'].sum()"

    monkeypatch.setattr(code_generator, "generate_code", fake_generate_code)
    monkeypatch.setattr(nodes, "generate_answer", lambda state: {"final_answer": "ok"})

    df = pd.DataFrame({"revenue": [10, 20, 30]})
    schema = {"columns": ["revenue"], "numeric_columns": ["revenue"], "categorical_columns": [], "date_columns": []}

    graph = build_agent_graph()
    state = {
        "dataframe": df,
        "schema": schema,
        "user_question": "total revenue",
        "mode": "analyze",
        "retry_count": 0,
    }

    final_state = None
    for ev in graph.stream(state, stream_mode="values"):
        final_state = ev

    assert final_state is not None
    assert final_state.get("error") is None
    assert final_state.get("execution_result") == 60
    assert isinstance(final_state.get("final_answer"), str)
