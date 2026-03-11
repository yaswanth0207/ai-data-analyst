"""
Streamlit UI for the AI Data Analyst.
Sidebar: file upload, load & analyze, dataset info, auto-insights, PDF export.
Tabs: Chat with Data, Auto Dashboard, Data Explorer.
"""
import io
import os
import sys

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# Ensure backend is on path when running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

@st.cache_data(show_spinner=False)
def _read_csv_cached(content: bytes) -> pd.DataFrame:
    """Cache CSV parsing to keep the UI responsive."""
    return pd.read_csv(io.BytesIO(content))


@st.cache_data(show_spinner=False, ttl=60)
def _get_chart_cached(chart_id: str) -> dict | None:
    """Cache chart fetches briefly to avoid repeated API calls while re-rendering."""
    return api_chart(chart_id)


def api_upload(filename: str, content: bytes) -> dict | None:
    """Upload CSV and return response with session_id and schema."""
    try:
        r = requests.post(
            f"{API_BASE}/upload",
            files={"file": (filename or "data.csv", content, "text/csv")},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Upload failed: {e}")
        return None


def api_analyze(question: str, mode: str, session_id: str) -> dict | None:
    """Send question to analyze endpoint."""
    try:
        r = requests.post(
            f"{API_BASE}/analyze",
            json={"question": question, "mode": mode, "session_id": session_id},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        return None


def api_insights(session_id: str) -> list:
    """Fetch auto-generated insights."""
    try:
        r = requests.get(f"{API_BASE}/insights", params={"session_id": session_id}, timeout=10)
        r.raise_for_status()
        return r.json().get("insights", [])
    except Exception:
        return []


def api_chart(chart_id: str) -> dict | None:
    """Fetch chart JSON by id."""
    try:
        r = requests.get(f"{API_BASE}/chart/{chart_id}", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def api_export(session_id: str) -> bytes | None:
    """Download PDF report."""
    try:
        r = requests.post(f"{API_BASE}/export", params={"session_id": session_id}, timeout=30)
        r.raise_for_status()
        return r.content
    except Exception as e:
        st.error(f"Export failed: {e}")
        return None


def api_history(session_id: str) -> list:
    """Fetch session history."""
    try:
        r = requests.get(f"{API_BASE}/history", params={"session_id": session_id}, timeout=10)
        r.raise_for_status()
        return r.json().get("history", [])
    except Exception:
        return []


st.set_page_config(page_title="AI Data Analyst", layout="wide")
st.title("AI Data Analyst")

# Session state
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "schema" not in st.session_state:
    st.session_state.schema = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "df" not in st.session_state:
    st.session_state.df = None

# Sidebar
with st.sidebar:
    st.header("Dataset")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if st.button("Load & Analyze"):
        if uploaded is not None:
            with st.status("Loading dataset…", expanded=False) as status:
                content = uploaded.read()
                status.update(label="Uploading CSV…", state="running")
                resp = api_upload(uploaded.name or "data.csv", content)
                if resp:
                    status.update(label="Parsing CSV locally…", state="running")
                    st.session_state.session_id = resp["session_id"]
                    st.session_state.schema = resp.get("schema_summary", {})
                    st.session_state.messages = []
                    st.session_state.df = _read_csv_cached(content)
                    status.update(label="Done.", state="complete")
                    st.success("Dataset loaded.")
                    st.rerun()
        else:
            st.warning("Select a CSV file first.")

    if st.session_state.session_id:
        sid = st.session_state.session_id
        schema = st.session_state.schema or {}
        st.metric("Rows", schema.get("row_count", "—"))
        st.metric("Columns", schema.get("column_count", "—"))
        st.caption("Columns: " + ", ".join((schema.get("columns") or [])[:8]) + ("..." if len(schema.get("columns") or []) > 8 else ""))

        st.subheader("Auto Insights")
        insights = api_insights(sid)
        for i, ins in enumerate(insights[:5], 1):
            st.markdown(f"{i}. {ins[:200]}{'...' if len(ins) > 200 else ''}")

        st.divider()
        if st.button("Download PDF Report"):
            pdf = api_export(sid)
            if pdf:
                st.session_state["report_pdf"] = pdf
        if st.session_state.get("report_pdf"):
            st.download_button("Save report.pdf", data=st.session_state["report_pdf"], file_name="report.pdf", mime="application/pdf", key="dl_pdf")

# Main area — Tabs
if not st.session_state.session_id:
    st.info("Upload a CSV and click **Load & Analyze** to start.")
    st.stop()

sid = st.session_state.session_id
df = st.session_state.df
schema = st.session_state.schema or {}

tab1, tab2, tab3 = st.tabs(["Chat with Data", "Auto Dashboard", "Data Explorer"])

# Tab 1: Chat with Data
with tab1:
    mode = st.selectbox(
        "Mode",
        ["analyze", "visualize", "summarize", "find_anomalies"],
        format_func=lambda x: {"analyze": "Analyze", "visualize": "Visualize", "summarize": "Summarize", "find_anomalies": "Find Anomalies"}[x],
    )
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("chart_id"):
                chart_data = _get_chart_cached(msg["chart_id"])
                if chart_data:
                    import plotly.graph_objects as go
                    fig = go.Figure(data=chart_data.get("data", []), layout=chart_data.get("layout", {}))
                    st.plotly_chart(fig, width="stretch")
            if msg.get("code"):
                with st.expander("Generated code", expanded=False):
                    st.code(msg["code"], language="python")
            if msg.get("similar"):
                st.caption("Similar question was asked: " + msg["similar"][:150])

    if prompt := st.chat_input("Ask a question about your data"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.status("Running analysis…", expanded=False) as status:
                status.update(label="Calling agent…", state="running")
                out = api_analyze(prompt, mode, sid)
            if out:
                st.markdown(out.get("answer", ""))
                chart_id = out.get("chart_id")
                if chart_id:
                    status.update(label="Fetching chart…", state="running")
                    chart_data = _get_chart_cached(chart_id)
                    if chart_data:
                        import plotly.graph_objects as go
                        fig = go.Figure(data=chart_data.get("data", []), layout=chart_data.get("layout", {}))
                        st.plotly_chart(fig, width="stretch")
                code = out.get("code", "")
                if code:
                    with st.expander("Generated code", expanded=False):
                        st.code(code, language="python")
                sim = out.get("similar_questions", [])
                if sim and len(sim) > 0:
                    st.caption("Similar question was asked: " + (sim[0].get("question", "")[:150] or ""))
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": out.get("answer", ""),
                    "code": code,
                    "chart_id": chart_id,
                    "similar": sim[0].get("question", "") if sim else None,
                })
                status.update(label="Done.", state="complete")

# Tab 2: Auto Dashboard
with tab2:
    if df is not None and len(df) > 0:
        numeric = df.select_dtypes(include="number").columns.tolist()
        cat = df.select_dtypes(include=["object", "category"]).columns.tolist()
        st.subheader("Auto-generated dashboard")
        c1, c2 = st.columns(2)
        with c1:
            if numeric:
                col = numeric[0]
                fig = px.histogram(df, x=col, title=f"Distribution of {col}")
                st.plotly_chart(fig, width="stretch")
            if len(numeric) >= 2:
                corr = df[numeric].corr()
                fig = px.imshow(corr, text_auto=".2f", title="Correlation heatmap")
                st.plotly_chart(fig, width="stretch")
        with c2:
            if cat and numeric:
                x_col = cat[0]
                y_col = numeric[0]
                agg = df.groupby(x_col)[y_col].sum().reset_index()
                fig = px.bar(agg.head(10), x=x_col, y=y_col, title=f"Top 10 by {y_col}")
                st.plotly_chart(fig, width="stretch")
            if len(numeric) >= 2:
                fig = px.scatter(df, x=numeric[0], y=numeric[1], title=f"{numeric[0]} vs {numeric[1]}")
                st.plotly_chart(fig, width="stretch")
        st.subheader("Key insights")
        insights = api_insights(sid)
        for i, ins in enumerate(insights[:5], 1):
            st.markdown(f"- **{i}.** {ins}")
    else:
        st.info("Load a dataset to see the auto dashboard.")

# Tab 3: Data Explorer
with tab3:
    if df is not None:
        st.dataframe(df, width="stretch")
        col_sel = st.multiselect("Columns to show", df.columns.tolist(), default=df.columns.tolist()[:5] or df.columns.tolist())
        if col_sel:
            st.dataframe(df[col_sel], width="stretch")
        st.subheader("Basic stats")
        st.dataframe(df.describe(), width="stretch")
        search = st.text_input("Filter rows (search)")
        if search:
            mask = df.astype(str).apply(lambda row: search.lower() in " ".join(row).lower(), axis=1)
            st.dataframe(df[mask], width="stretch")
    else:
        st.info("Load a dataset to explore.")
