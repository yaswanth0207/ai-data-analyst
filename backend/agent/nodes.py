"""
LangGraph node functions: load_data, route_question, generate_code, execute_code, check_error,
fix_code, generate_answer, generate_insights.
"""
import os
import re
from difflib import get_close_matches
from typing import Any, Literal

import numpy as np
import pandas as pd
from langchain_ollama import ChatOllama

from backend.agent.state import AgentState
from backend.analyst import code_executor
from backend.analyst import code_generator
from backend.analyst import insight_engine
from backend.analyst import schema_inspector


def load_data(state: AgentState) -> dict[str, Any]:
    """Read CSV, build schema, optionally run insight engine. Requires uploaded_file_path."""
    path = state.get("uploaded_file_path") or ""
    if not path or not os.path.isfile(path):
        return {"error": "No valid file path provided."}
    try:
        df, schema = schema_inspector.inspect_csv(path)
        insights = insight_engine.generate_insights(df, schema)
        return {
            "dataframe": df,
            "schema": schema,
            "insights": insights,
            "error": None,
        }
    except Exception as e:
        return {"error": str(e)}


def route_question(state: AgentState) -> dict[str, Any]:
    """Set mode from user question: analyze / visualize / summarize / find_anomalies."""
    q = (state.get("user_question") or "").lower()
    if any(w in q for w in ["chart", "plot", "graph", "visualize", "visualization"]):
        mode = "visualize"
    elif any(w in q for w in ["summarize", "summary", "overview", "describe the data"]):
        mode = "summarize"
    elif any(w in q for w in ["anomal", "outlier", "unusual", "exception"]):
        mode = "find_anomalies"
    else:
        mode = "analyze"
    return {"mode": mode}


def generate_code_node(state: AgentState) -> dict[str, Any]:
    """Generate pandas/plotly code from user question and schema."""
    schema = state.get("schema") or {}
    question = state.get("user_question") or ""
    mode = state.get("mode") or "analyze"
    try:
        code = code_generator.generate_code(question, schema, mode)
        return {"generated_code": code, "error": None}
    except Exception as e:
        return {"generated_code": "", "error": str(e)}


def execute_code_node(state: AgentState) -> dict[str, Any]:
    """Run generated code in sandbox. Requires dataframe and generated_code."""
    df = state.get("dataframe")
    code = state.get("generated_code") or ""
    if df is None:
        return {"execution_result": None, "chart": None, "error": "No dataframe loaded."}
    timeout = int(os.getenv("CODE_TIMEOUT_SECONDS", "30"))
    out = code_executor.execute_code(code, df, timeout_seconds=timeout)
    result = out.get("result")
    fig = out.get("fig")
    err = out.get("error")
    # Keep result as-is; API will serialize DataFrame to dict/list for JSON
    return {
        "execution_result": result,
        "chart": fig,
        "error": err,
    }


def _extract_missing_column(error_text: str) -> str | None:
    """Best-effort extraction of a missing column name from common pandas errors."""
    m = re.search(r"KeyError:\\s*'([^']+)'", error_text)
    if m:
        return m.group(1)
    m = re.search(r"\\[\\s*'([^']+)'\\s*\\]\\s+not\\s+in\\s+index", error_text)
    if m:
        return m.group(1)
    m = re.search(r"NameError:\\s*name\\s+'([^']+)'\\s+is\\s+not\\s+defined", error_text)
    if m:
        return m.group(1)
    return None


def _friendly_error_message(error_text: str, schema: dict[str, Any]) -> str:
    """Turn raw execution errors into user-friendly messages with suggestions."""
    cols: list[str] = list(schema.get("columns") or [])
    missing = _extract_missing_column(error_text)
    if missing and cols:
        suggestions = get_close_matches(missing, cols, n=3, cutoff=0.5)
        if suggestions:
            return f"Column '{missing}' not found. Did you mean: {', '.join([repr(s) for s in suggestions])}?"
        return (
            f"Column '{missing}' not found. Available columns include: "
            f"{', '.join([repr(c) for c in cols[:10]])}{'...' if len(cols) > 10 else ''}"
        )

    if "Import statements are not allowed" in error_text:
        return (
            "The generated code tried to import a library. Imports are blocked in the sandbox; "
            "use the provided variables: df, pd, np, px, go."
        )

    if "Forbidden name" in error_text or "Forbidden call" in error_text:
        return (
            "The generated code attempted a disallowed operation in the sandbox. "
            "Try rephrasing the question to only use data operations (filters, groupby, aggregations, plotly charts)."
        )

    if "timed out" in error_text.lower():
        return (
            "That analysis took too long to run (timeout). Try asking for a smaller result "
            "(e.g., top 20 rows) or a simpler aggregation."
        )

    return f"The analysis failed while running generated code. Error: {error_text}"


def _deterministic_anomaly_answer(df: pd.DataFrame, schema: dict[str, Any]) -> str:
    """Deterministic explanation for find_anomalies mode (z-score > 3)."""
    numeric = schema.get("numeric_columns") or df.select_dtypes(include="number").columns.tolist()
    target = None
    for cand in ["revenue", "sales", "amount", "total", "value"]:
        if cand in numeric:
            target = cand
            break
    target = target or (numeric[0] if numeric else None)
    if target is None:
        return "No numeric columns found, so I can't detect outliers."

    s = pd.to_numeric(df[target], errors="coerce")
    mean_val = float(s.mean())
    std_val = float(s.std())
    if std_val == 0 or np.isnan(std_val):
        return f"No variation in '{target}', so I can't compute outliers."

    z = (s - mean_val) / std_val
    out_mask = z.abs() > 3
    outliers = df.loc[out_mask].copy()
    outliers["z_score"] = z[out_mask].round(2)
    outliers = outliers.sort_values("z_score", ascending=False)

    n_out = int(outliers.shape[0])
    if n_out == 0:
        return f"No outliers detected in '{target}' using z-score > 3 (mean={mean_val:.2f}, std={std_val:.2f})."

    cols_to_show = [c for c in ["order_id", "order_date", "product", "region", target, "z_score"] if c in outliers.columns]
    top = outliers[cols_to_show].head(3)
    lines = [
        f"Detected {n_out} outlier(s) in '{target}' using z-score > 3 (mean={mean_val:.2f}, std={std_val:.2f}).",
        "Top outlier(s):",
    ]
    for _, row in top.iterrows():
        parts = [f"{c}={row[c]}" for c in cols_to_show]
        lines.append("- " + ", ".join(parts))
    return "\n".join(lines)


def check_error(state: AgentState) -> Literal["generate_answer", "fix_code"]:
    """Route: if error and retry_count < 2 -> fix_code, else -> generate_answer."""
    err = state.get("error")
    retry = state.get("retry_count") or 0
    max_retries = int(os.getenv("MAX_RETRIES", "2"))
    if err and retry < max_retries:
        return "fix_code"
    return "generate_answer"


def fix_code(state: AgentState) -> dict[str, Any]:
    """Regenerate code with previous error message for self-healing."""
    schema = state.get("schema") or {}
    question = state.get("user_question") or ""
    mode = state.get("mode") or "analyze"
    prev_error = state.get("error") or ""
    prev_code = state.get("generated_code") or ""
    retry = state.get("retry_count") or 0
    client = ChatOllama(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        model=os.getenv("OLLAMA_MODEL", "llama3.2"),
        temperature=0.1,
    )
    from langchain_core.messages import HumanMessage, SystemMessage

    system = "You are a Python data analyst. Fix the following code. Output ONLY executable Python code, no markdown. DataFrame is 'df'. Store result in 'result', plotly figure in 'fig' if needed."
    user = f"Previous code:\n{prev_code}\n\nError: {prev_error}\n\nUser question: {question}\n\nProvide corrected code only."
    resp = client.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    content = resp.content if hasattr(resp, "content") else str(resp)
    import re
    match = re.search(r"```(?:python)?\s*([\s\S]*?)```", content.strip())
    new_code = match.group(1).strip() if match else content.strip()
    return {
        "generated_code": new_code,
        "retry_count": retry + 1,
        "error": None,
    }


def generate_answer(state: AgentState) -> dict[str, Any]:
    """Use LLM to explain execution result in plain English."""
    question = state.get("user_question") or ""
    result = state.get("execution_result")
    err = state.get("error")
    schema = state.get("schema") or {}
    code = state.get("generated_code") or ""
    mode = state.get("mode") or "analyze"
    df = state.get("dataframe")
    client = ChatOllama(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        model=os.getenv("OLLAMA_MODEL", "llama3.2"),
        temperature=0.2,
    )
    from langchain_core.messages import HumanMessage, SystemMessage

    if err:
        # Deterministic, user-friendly errors with suggestions.
        return {"final_answer": _friendly_error_message(err, schema)}
    if mode == "find_anomalies" and df is not None:
        # Deterministic anomaly explanation (avoid LLM hallucinations).
        return {"final_answer": _deterministic_anomaly_answer(df, schema)}
    prompt = f"The user asked: {question}\n\nCode result (or description): {result}\n\nProvide a clear, plain-English answer summarizing the result. If there is a table/chart, describe what it shows."
    resp = client.invoke(
        [
            SystemMessage(content="You are a helpful data analyst. Answer in plain English, concisely."),
            HumanMessage(content=prompt),
        ]
    )
    answer = resp.content if hasattr(resp, "content") else str(resp)
    return {"final_answer": answer}


def generate_insights_node(state: AgentState) -> dict[str, Any]:
    """Compute auto-insights from current dataframe and schema. Used after load_data."""
    df = state.get("dataframe")
    schema = state.get("schema") or {}
    if df is None:
        return {"insights": []}
    return {"insights": insight_engine.generate_insights(df, schema)}
