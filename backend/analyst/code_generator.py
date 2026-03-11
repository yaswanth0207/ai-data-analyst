"""
Code generator: uses Ollama LLM to generate pandas/plotly code from user question + schema.
"""
import os
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama


def get_ollama_client() -> ChatOllama:
    """Build Ollama chat client from env."""
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3.2")
    return ChatOllama(base_url=base_url, model=model, temperature=0.1)


SYSTEM_PROMPT = """You are a Python data analyst. Generate ONLY executable Python code.
- The DataFrame is already loaded as 'df'. Do not load or read files.
- Store your result in a variable called 'result' (e.g. result = df.head(10), or result = df.groupby(...).sum()).
- For charts/plots, also create a plotly figure and store it in variable 'fig' (e.g. fig = px.bar(...)).
- You may use: pandas (as pd), numpy (as np), plotly.express (as px), plotly.graph_objects (as go).
- No other imports. No explanations. No markdown. Output only raw code.
- If the request is impossible with the given schema, still output code that produces a clear result variable explaining why (e.g. result = "Column X not found in schema.").
"""


def _schema_context(schema: dict[str, Any]) -> str:
    """Format schema dict as string for LLM context."""
    parts = [
        f"Columns: {schema.get('columns', [])}",
        f"Dtypes: {schema.get('dtypes', {})}",
        f"Numeric: {schema.get('numeric_columns', [])}",
        f"Categorical: {schema.get('categorical_columns', [])}",
        f"Date: {schema.get('date_columns', [])}",
        f"Sample rows: {schema.get('sample_rows', [])}",
    ]
    return "\n".join(parts)


def _extract_code(text: str) -> str:
    """Extract code block from LLM response (markdown or raw)."""
    text = text.strip()
    # Remove markdown code fences
    match = re.search(r"```(?:python)?\s*([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()
    return text


def generate_code(
    user_question: str,
    schema: dict[str, Any],
    mode: str,
) -> str:
    """
    Generate pandas/plotly code from user question.
    Returns clean executable code string; caller injects df and runs in sandbox.
    """
    client = get_ollama_client()
    schema_str = _schema_context(schema)
    mode_hint = {
        "analyze": "Answer the question with a table or aggregation. Store in 'result'.",
        "visualize": "Create a plotly chart. Store figure in 'fig'. Also set result = 'Chart generated.' or similar.",
        "summarize": "Provide dataset overview: shape, dtypes, describe(), sample. Store in 'result'.",
        "find_anomalies": "Find outliers (e.g. values > 3 std from mean). Store findings in 'result', optionally 'fig'.",
    }.get(mode, "Answer the question. Store in 'result', use 'fig' for charts.")

    user_message = f"""Schema and data context:
{schema_str}

Mode: {mode}. {mode_hint}

User question: {user_question}

Generate only Python code. No markdown, no explanation."""

    response = client.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]
    )
    content = response.content if hasattr(response, "content") else str(response)
    return _extract_code(content)
