"""
Safe sandbox for executing generated pandas/plotly code.
Restricts builtins and allows only pandas, numpy, plotly. Timeout 30s.
"""
import signal
from io import StringIO
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# Forbidden names that cannot be used in user code
FORBIDDEN = frozenset(
    {
        "os", "sys", "subprocess", "open", "eval", "exec", "compile",
        "input", "file", "reload", "breakpoint", "globals", "locals",
        "getattr", "setattr", "delattr", "vars", "dir", "import",
        "__import__", "exit", "quit", "help", "license", "copyright",
        "memoryview", "buffer", "execfile", "raw_input", "assert",
    }
)


def _safe_builtins() -> dict[str, Any]:
    """Minimal safe builtins for exec()."""
    safe = {
        "abs": abs, "all": all, "any": any, "bin": bin, "bool": bool,
        "dict": dict, "enumerate": enumerate, "filter": filter,
        "float": float, "int": int, "iter": iter, "len": len,
        "list": list, "map": map, "max": max, "min": min,
        "next": next, "range": range, "round": round, "set": set,
        "sorted": sorted, "str": str, "sum": sum, "tuple": tuple,
        "zip": zip, "True": True, "False": False, "None": None,
        "print": print,
    }
    return safe


def _strip_allowed_imports(code: str) -> str:
    """Remove lines that only import pandas, numpy, or plotly (we inject these in namespace)."""
    allowed_prefixes = ("import pandas", "import numpy", "import plotly", "from pandas", "from numpy", "from plotly")
    lines = []
    for line in code.splitlines():
        stripped = line.strip()
        if any(stripped.startswith(p) for p in allowed_prefixes):
            continue
        lines.append(line)
    return "\n".join(lines)


def _check_code_safety(code: str) -> None:
    """Raise ValueError if code uses forbidden names or any imports (after stripping allowed)."""
    code = _strip_allowed_imports(code)
    import ast
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ValueError(f"Invalid Python syntax: {e}") from e
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if node.id in FORBIDDEN:
                raise ValueError(f"Forbidden name: {node.id}")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN:
                raise ValueError(f"Forbidden call: {node.func.id}")
        if isinstance(node, ast.Import):
            raise ValueError("Import statements are not allowed (use pd, np, px, go from namespace)")
        if isinstance(node, ast.ImportFrom):
            raise ValueError("Import statements are not allowed (use pd, np, px, go from namespace)")


def execute_code(
    code: str,
    df: pd.DataFrame,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    """
    Execute code in restricted namespace with df, pandas, numpy, plotly.
    Returns dict: result, fig, error, stdout.
    """
    code = _strip_allowed_imports(code)
    try:
        _check_code_safety(code)
    except Exception as e:
        return {
            "result": None,
            "fig": None,
            "error": f"{type(e).__name__}: {e}",
            "stdout": "",
        }
    namespace: dict[str, Any] = {
        "df": df,
        "pd": pd,
        "np": np,
        "px": px,
        "go": go,
        "result": None,
        "fig": None,
    }
    namespace.update(_safe_builtins())
    stdout_capture = StringIO()

    def timeout_handler(signum, frame):
        raise TimeoutError("Code execution timed out")

    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        try:
            # Redirect print to capture (exec uses namespace["print"])
            def capturing_print(*args, **kwargs):
                kwargs.setdefault("file", stdout_capture)
                print(*args, **kwargs)
            namespace["print"] = lambda *a, **k: capturing_print(*a, **k)
            exec(code, namespace)  # use stripped code (no import lines)
        finally:
            signal.alarm(0)
    except TimeoutError as e:
        return {
            "result": None,
            "fig": None,
            "error": str(e),
            "stdout": stdout_capture.getvalue(),
        }
    except Exception as e:
        return {
            "result": None,
            "fig": None,
            "error": f"{type(e).__name__}: {e}",
            "stdout": stdout_capture.getvalue(),
        }

    return {
        "result": namespace.get("result"),
        "fig": namespace.get("fig"),
        "error": None,
        "stdout": stdout_capture.getvalue(),
    }
