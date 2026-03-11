import pandas as pd

from backend.analyst.schema_inspector import inspect_csv


def test_inspect_csv_extracts_schema(tmp_path):
    p = tmp_path / "tiny.csv"
    p.write_text(
        "a,b,date\n"
        "1,x,2024-01-01\n"
        "2,y,2024-01-02\n"
    )

    df, schema = inspect_csv(str(p))

    assert isinstance(df, pd.DataFrame)
    assert schema["row_count"] == 2
    assert schema["column_count"] == 3
    assert schema["columns"] == ["a", "b", "date"]
    assert "a" in schema["numeric_columns"]
    assert "b" in schema["categorical_columns"]
    # date-like string columns are detected
    assert "date" in schema["date_columns"]
