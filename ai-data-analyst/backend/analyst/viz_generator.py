"""
Visualization generator: map user intent to chart type and generate plotly code/figures.
Supports bar, line, scatter, histogram, heatmap, pie. Auto-detect best chart from question.
"""
from typing import Any, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# Keywords that suggest chart type
CHART_HINTS = {
    "bar": ["bar", "bars", "compare", "top n", "ranking", "by category", "count by"],
    "line": ["trend", "over time", "time series", "monthly", "daily", "growth", "line"],
    "scatter": ["scatter", "correlation", "relationship", "vs ", " versus ", "against"],
    "histogram": ["distribution", "histogram", "frequency", "how many", "range"],
    "heatmap": ["heatmap", "matrix", "correlation matrix", "cross-tab"],
    "pie": ["pie", "proportion", "share", "percentage of", "breakdown"],
}


def _detect_chart_type(question: str, schema: dict[str, Any]) -> str:
    """Infer best chart type from user question (and schema if needed)."""
    q = question.lower().strip()
    for chart_type, keywords in CHART_HINTS.items():
        if any(kw in q for kw in keywords):
            return chart_type
    # Default: trend -> line, top N -> bar, distribution -> histogram
    if any(w in q for w in ["trend", "over time", "monthly", "sales trend"]):
        return "line"
    if any(w in q for w in ["top ", "best ", "most ", "highest", "lowest"]):
        return "bar"
    if any(w in q for w in ["distribution", "spread", "range"]):
        return "histogram"
    return "bar"


def figure_to_json(fig: go.Figure) -> dict[str, Any]:
    """Convert plotly figure to JSON for frontend (Streamlit can use plotly_chart with fig or dict)."""
    return fig.to_dict()


def generate_chart(
    df: pd.DataFrame,
    question: str,
    schema: dict[str, Any],
) -> Optional[go.Figure]:
    """
    Generate a plotly figure based on question and schema.
    Returns None if data/schema insufficient; otherwise the figure.
    """
    chart_type = _detect_chart_type(question, schema)
    numeric = schema.get("numeric_columns") or []
    categorical = schema.get("categorical_columns") or []
    date_cols = schema.get("date_columns") or []
    columns = schema.get("columns") or list(df.columns)

    if len(columns) == 0:
        return None

    try:
        if chart_type == "bar":
            x_col = categorical[0] if categorical else columns[0]
            y_col = numeric[0] if numeric and numeric[0] != x_col else None
            if y_col and y_col in df.columns:
                agg = df.groupby(x_col)[y_col].sum().reset_index()
            else:
                agg = df.groupby(x_col).size().reset_index(name="count")
                y_col = "count"
            return px.bar(agg, x=x_col, y=y_col, title=question[:60])

        if chart_type == "line":
            x_col = date_cols[0] if date_cols else (categorical[0] if categorical else columns[0])
            y_col = numeric[0] if numeric and numeric[0] != x_col else None
            if y_col and y_col in df.columns:
                line_df = df.groupby(x_col)[y_col].sum().reset_index()
            else:
                line_df = df.groupby(x_col).size().reset_index(name="count")
                y_col = "count"
            return px.line(line_df, x=x_col, y=y_col, title=question[:60])

        if chart_type == "scatter":
            x_col = numeric[0] if numeric else columns[0]
            y_col = numeric[1] if len(numeric) > 1 else (columns[1] if len(columns) > 1 else columns[0])
            return px.scatter(df, x=x_col, y=y_col, title=question[:60])

        if chart_type == "histogram":
            col = numeric[0] if numeric else columns[0]
            return px.histogram(df, x=col, title=question[:60])

        if chart_type == "heatmap":
            num_cols = numeric[:5] or [c for c in columns if df[c].dtype in ["int64", "float64"]][:5]
            if len(num_cols) < 2:
                corr = df.select_dtypes(include="number").corr()
            else:
                corr = df[num_cols].corr()
            return px.imshow(corr, text_auto=".2f", title="Correlation heatmap")

        if chart_type == "pie":
            x_col = categorical[0] if categorical else columns[0]
            counts = df[x_col].value_counts().reset_index()
            counts.columns = ["name", "value"]
            return px.pie(counts, names="name", values="value", title=question[:60])

    except Exception:
        return None
    return None
