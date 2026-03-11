"""
LangGraph agent state schema for the AI Data Analyst.
All fields used across nodes for data loading, code gen, execution, and answers.
"""
from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """State passed between LangGraph nodes."""

    # Data source
    uploaded_file_path: str
    dataframe: Any  # pandas DataFrame
    schema: dict  # column names, types, sample values

    # User input
    user_question: str

    # Code execution
    generated_code: str
    execution_result: str
    chart: Any  # plotly figure
    error: str

    # Outputs
    insights: list[str]
    final_answer: str

    # Control
    mode: str  # analyze | visualize | summarize | find_anomalies
    retry_count: int  # for self-healing loop
