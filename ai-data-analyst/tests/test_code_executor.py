import pandas as pd

from backend.analyst.code_executor import execute_code


def test_executor_blocks_open_and_os(df=None):
    df = pd.DataFrame({"x": [1, 2, 3]})
    out = execute_code("result = open('x','w')", df)
    assert out["error"]
    assert "Forbidden" in out["error"] or "not allowed" in out["error"]


def test_executor_strips_allowed_imports():
    df = pd.DataFrame({"x": [1, 2, 3]})
    code = "import pandas as pd\nresult = df['x'].sum()"
    out = execute_code(code, df)
    assert out["error"] is None
    assert out["result"] == 6
