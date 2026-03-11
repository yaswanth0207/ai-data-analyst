"""
Insight engine: auto-generate insights from dataset without user asking.
Statistics, correlations, missing data, top/bottom N, trends, anomalies.
"""
from typing import Any

import numpy as np
import pandas as pd


def generate_insights(df: pd.DataFrame, schema: dict[str, Any]) -> list[str]:
    """
    Auto-generate a list of insight strings for the dataset.
    """
    insights: list[str] = []
    numeric = schema.get("numeric_columns") or df.select_dtypes(include="number").columns.tolist()
    categorical = schema.get("categorical_columns") or df.select_dtypes(include=["object", "category"]).columns.tolist()
    date_cols = schema.get("date_columns") or []
    row_count = len(df)

    # Top 5 statistics (mean, median, outliers)
    for col in numeric[:5]:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        mean_val = s.mean()
        median_val = s.median()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_outliers = ((s < low) | (s > high)).sum()
        insights.append(
            f"**{col}**: mean={mean_val:.2f}, median={median_val:.2f}"
            + (f"; {n_outliers} potential outliers (IQR method)." if n_outliers > 0 else ".")
        )

    # Correlation between numeric columns
    if len(numeric) >= 2:
        corr = df[numeric].corr()
        # Top pair by absolute correlation (excluding diagonal)
        best_pair = None
        best_abs = -1
        for i in range(len(numeric)):
            for j in range(len(numeric)):
                if i != j:
                    v = abs(corr.iloc[i, j])
                    if v > best_abs and not np.isnan(v):
                        best_abs = v
                        best_pair = (numeric[i], numeric[j], corr.iloc[i, j])
        if best_pair:
            insights.append(
                f"Strongest correlation: **{best_pair[0]}** and **{best_pair[1]}** (r={best_pair[2]:.2f})."
            )

    # Missing data patterns
    nulls = df.isnull().sum()
    null_cols = nulls[nulls > 0]
    if len(null_cols) > 0:
        total_missing = null_cols.sum()
        pct = 100 * total_missing / (row_count * len(df.columns))
        insights.append(
            f"Missing data: {total_missing} values ({pct:.1f}%) across {len(null_cols)} columns: "
            + ", ".join(null_cols.index.tolist()[:5]) + ("..." if len(null_cols) > 5 else ".")
        )
    else:
        insights.append("No missing values in the dataset.")

    # Top/bottom N values (for first numeric column if present)
    if numeric:
        col = numeric[0]
        top_val = df[col].max()
        bottom_val = df[col].min()
        insights.append(f"**{col}** range: min={bottom_val:.2f}, max={top_val:.2f}.")

    # Trend detection in date columns
    for col in date_cols[:1]:
        try:
            ts = pd.to_datetime(df[col], errors="coerce").dropna()
            if len(ts) < 2:
                continue
            ts = ts.sort_values()
            first, last = ts.iloc[0], ts.iloc[-1]
            insights.append(f"Date range in **{col}**: {first} to {last}.")
        except Exception:
            pass

    # Anomaly detection (values > 3 std from mean)
    for col in numeric[:3]:
        s = df[col].dropna()
        if len(s) < 4:
            continue
        mean_val = s.mean()
        std_val = s.std()
        if std_val == 0:
            continue
        z = np.abs((s - mean_val) / std_val)
        anomalies = (z > 3).sum()
        if anomalies > 0:
            insights.append(f"**{col}**: {anomalies} values are more than 3 standard deviations from the mean (potential anomalies).")

    return insights[:10]  # Cap at 10 insights
