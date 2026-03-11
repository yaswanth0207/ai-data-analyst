"""
Schema inspector: read CSV, extract column metadata, and build structured schema for LLM context.
"""
from pathlib import Path
from typing import Any

import pandas as pd


def inspect_csv(file_path: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Read uploaded CSV into DataFrame and build schema dict.
    Returns (dataframe, schema_dict) for use by agent and code generator.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)

    # Column metadata
    columns = list(df.columns)
    dtypes = {col: str(dt) for col, dt in df.dtypes.items()}
    null_counts = df.isnull().sum().to_dict()
    unique_counts = df.nunique().to_dict()

    # Sample rows (3) as list of dicts for LLM context
    sample_rows = df.head(3).fillna("").to_dict(orient="records")

    # Classify columns
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    date_cols: list[str] = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_cols.append(col)
        elif df[col].dtype == object:
            # Try parsing string columns that look like dates
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().sum() > 0:
                    date_cols.append(col)
            except Exception:
                pass

    schema = {
        "columns": columns,
        "dtypes": dtypes,
        "null_counts": null_counts,
        "unique_counts": unique_counts,
        "sample_rows": sample_rows,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "date_columns": date_cols,
        "row_count": len(df),
        "column_count": len(columns),
    }
    return df, schema
