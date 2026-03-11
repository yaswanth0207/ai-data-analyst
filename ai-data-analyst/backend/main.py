"""
FastAPI entry point for the AI Data Analyst backend.
Endpoints: upload, analyze, insights, chart, export, history.
"""
import os
import uuid
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.agent.graph import agent_graph
from backend.agent.state import AgentState
from backend.memory.query_memory import add_query, search_similar
from backend.utils.file_handler import save_uploaded_csv

# In-memory session store: session_id -> { dataframe, schema, insights, file_path, history }
_sessions: dict[str, dict[str, Any]] = {}
# Chart store: chart_id -> plotly figure dict (for GET /chart/{id})
_charts: dict[str, dict[str, Any]] = {}

app = FastAPI(title="AI Data Analyst API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    question: str
    mode: str = "analyze"  # analyze | visualize | summarize | find_anomalies
    session_id: str | None = None


class AnalyzeResponse(BaseModel):
    session_id: str
    answer: str
    code: str
    execution_result: Any
    chart_id: str | None = None
    similar_questions: list[dict[str, Any]] = []


def _get_or_create_session(session_id: str | None) -> str:
    if session_id and session_id in _sessions:
        return session_id
    new_id = str(uuid.uuid4())
    _sessions[new_id] = {"dataframe": None, "schema": None, "insights": [], "file_path": None, "history": []}
    return new_id


def _serialize_result(result: Any) -> Any:
    """Convert result to JSON-serializable form."""
    if result is None:
        return None
    if isinstance(result, pd.DataFrame):
        return result.head(100).to_dict(orient="records")
    if hasattr(result, "to_dict") and callable(getattr(result, "to_dict")):
        return result.to_dict()
    return result


@app.post("/upload")
async def upload_csv(file: UploadFile = File(...), session_id: str | None = None):
    """Accept CSV upload, save to temp, run schema inspection, return schema summary and session_id."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Only CSV files are allowed")
    sid = _get_or_create_session(session_id)
    path = save_uploaded_csv(file)
    if not path:
        raise HTTPException(400, "Failed to save file")
    try:
        from backend.analyst.schema_inspector import inspect_csv
        from backend.analyst.insight_engine import generate_insights
        df, schema = inspect_csv(path)
        insights = generate_insights(df, schema)
    except Exception as e:
        raise HTTPException(400, str(e))
    _sessions[sid]["dataframe"] = df
    _sessions[sid]["schema"] = schema
    _sessions[sid]["insights"] = insights
    _sessions[sid]["file_path"] = path
    return {
        "session_id": sid,
        "schema_summary": {
            "row_count": schema["row_count"],
            "column_count": schema["column_count"],
            "columns": schema["columns"],
            "numeric_columns": schema.get("numeric_columns", []),
            "categorical_columns": schema.get("categorical_columns", []),
        },
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    """Run the agent on the question; return answer, code, optional chart_id, and similar past questions."""
    sid = _get_or_create_session(req.session_id)
    session = _sessions.get(sid)
    if not session or session.get("dataframe") is None:
        raise HTTPException(400, "Upload a CSV first (POST /upload)")
    df = session["dataframe"]
    schema = session["schema"]
    config = {"configurable": {"thread_id": sid}}
    initial_state: AgentState = {
        "uploaded_file_path": session.get("file_path") or "",
        "dataframe": df,
        "schema": schema,
        "user_question": req.question,
        "mode": req.mode,
        "retry_count": 0,
    }
    # Search similar past questions before running
    similar = search_similar(req.question, n_results=3)
    # Run graph (from route_question since we have data)
    try:
        final_state = None
        for event in agent_graph.stream(initial_state, config=config, stream_mode="values"):
            final_state = event
        if not final_state:
            raise HTTPException(500, "Agent produced no final state")
    except Exception as e:
        raise HTTPException(500, f"Agent error: {e}")
    answer = final_state.get("final_answer") or ""
    code = final_state.get("generated_code") or ""
    exec_result = final_state.get("execution_result")
    chart = final_state.get("chart")
    chart_id = None
    if chart is not None and hasattr(chart, "to_dict"):
        chart_id = str(uuid.uuid4())
        _charts[chart_id] = chart.to_dict()
    # Store in history and in ChromaDB
    session["history"] = session.get("history") or []
    session["history"].append({
        "question": req.question,
        "answer": answer,
        "code": code,
        "execution_result": _serialize_result(exec_result),
        "chart_id": chart_id,
    })
    add_query(req.question, answer, metadata={"session_id": sid})
    return AnalyzeResponse(
        session_id=sid,
        answer=answer,
        code=code,
        execution_result=_serialize_result(exec_result),
        chart_id=chart_id,
        similar_questions=similar,
    )


@app.get("/insights")
async def get_insights(session_id: str):
    """Return auto-generated insights for the loaded dataset."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return {"insights": session.get("insights") or []}


@app.get("/chart/{chart_id}")
async def get_chart(chart_id: str):
    """Return plotly chart as JSON for the frontend."""
    if chart_id not in _charts:
        raise HTTPException(404, "Chart not found")
    return _charts[chart_id]


@app.post("/export")
async def export_report(session_id: str):
    """Generate and return PDF report of the session (summary, Q&A, charts, insights)."""
    from backend.export.report_generator import generate_report_pdf
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    try:
        pdf_path = generate_report_pdf(session_id, session, _charts)
        from fastapi.responses import FileResponse
        return FileResponse(pdf_path, media_type="application/pdf", filename="report.pdf")
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/history")
async def get_history(session_id: str):
    """Return past questions and answers for the session."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return {"history": session.get("history") or []}
