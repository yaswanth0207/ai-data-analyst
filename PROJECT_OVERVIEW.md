# AI Data Analyst Agent — Project Notes

This document is a high-signal walkthrough of the **AI Data Analyst Agent** project: what it does, how it’s built, and the design choices/trade-offs you can explain confidently.

## What the project does

- Users upload a **CSV**.
- They ask questions in plain English (e.g., “top 5 products by revenue”, “monthly trend”, “find outliers”).
- The system:
  - inspects schema,
  - generates code,
  - executes it in a restricted sandbox,
  - returns an answer + evidence (table + chart) + generated code,
  - stores Q&A in vector memory (ChromaDB),
  - exports a PDF session report.

## Tech stack (and why)

- **FastAPI**: simple HTTP API for upload, analysis, history, export.
- **Streamlit**: fast UI iteration; good for data apps (tables/charts).
- **LangGraph**: explicit multi-step agent workflow (nodes, routing, retries).
- **Ollama (local)**: runs LLM locally (privacy + no API keys).
- **pandas / numpy**: dataframe transforms + numeric operations.
- **plotly**: interactive charts for the UI.
- **ChromaDB**: similarity search for “similar past questions”.
- **reportlab**: generate PDF reports.

## Architecture (how requests flow)

```mermaid
graph LR
  U[User] --> S[Streamlit UI]
  S -->|upload| API[FastAPI]
  API --> SI[Schema Inspector]
  API -->|analyze| LG[LangGraph Agent]
  LG --> R[Route mode: analyze/visualize/summarize/anomaly]
  R --> CG[Code Generator (Ollama or deterministic)]
  CG --> EX[Sandbox Executor (restricted exec)]
  EX --> ANS[Answer (LLM or deterministic)]
  API --> MEM[ChromaDB similar queries]
  API --> PDF[ReportLab PDF]
  API --> S
```

## Core agent workflow (LangGraph)

The agent is a **graph** rather than a single prompt. It makes behavior predictable and debuggable.

High-level node sequence:

- **load_data**: load CSV → build `schema` → compute auto-insights
- **route_question**: decide mode (`analyze`, `visualize`, `summarize`, `find_anomalies`)
- **generate_code**: produce pandas/plotly code (LLM), except some modes can be deterministic
- **execute_code**: run generated code in sandbox
- **fix_code**: on error, ask LLM to correct code and retry (bounded retries)
- **generate_answer**: explain results (LLM), with deterministic paths for safe/factual responses

Key design choices:

- **Self-healing loop**: bounded retries prevent infinite loops and keep UX smooth.
- **No persistent checkpointer**: LangGraph checkpointers msgpack-serialize state; pandas DataFrames aren’t serializable → graph runs in-memory per request.

## State management

- API keeps an in-memory session map:
  - dataframe, schema, insights, history, file path
- Each `/analyze` run uses the session’s df + schema.

Trade-off:

- **Pros**: very fast; no DB required for MVP.
- **Cons**: loses sessions on server restart; not horizontally scalable.

How you’d scale it:

- store session metadata and file path in Redis/Postgres,
- store data in object storage (S3 / local volume),
- keep only lightweight pointers in memory.

## Code generation strategy

### Normal modes (analyze/visualize/summarize)

The LLM is prompted to output **code only**:

- DataFrame is always `df`
- store output in `result`
- store plotly chart in `fig`
- no explanations in the code

### Deterministic “find anomalies”

We switched `find_anomalies` to a deterministic implementation because LLMs can hallucinate:

- z-score outlier detection (abs(z) > 3)
- always returns:
  - `result` as an outliers table (with `z_score`)
  - `fig` as a plotly box plot
- and a deterministic explanation describing mean/std and top outliers

This is a good example of a hybrid system: **LLM where flexible, deterministic where correctness matters**.

## Sandbox execution & security posture

Generated code executes via `exec()` inside a restricted namespace.

**Allowed in namespace**:

- `df`, `pd`, `np`, `px`, `go`
- minimal safe builtins (e.g., `len`, `sum`, `sorted`, `range`, etc.)

**Blocked**:

- imports (imports of pandas/numpy/plotly are stripped because those are already provided),
- file/network operations (`open`, `os`, `sys`, `subprocess`, etc.)

Timeout:

- execution is bounded (alarm-based timeout).

Talking points:

- sandboxing is hard to make “perfect” in Python; this is a pragmatic MVP.
- next step: run execution in a separate process/container with OS-level sandboxing.

## Evidence-first answers

UI shows:

- the natural-language answer,
- the generated code (collapsed),
- table output (from `result`),
- interactive chart (from `fig`).

This reduces “trust gaps” and makes debugging easier.

## Vector memory (ChromaDB)

For each Q&A:

- store question text
- store answer snippet in metadata

On new question:

- search similar past questions
- show “Similar question was asked: …”

Trade-off:

- helpful for discovery, but similarity search depends on embeddings/model quality.

## PDF export

Report includes:

- dataset summary
- key insights
- Q&A history

Note:

- plotly figures aren’t rendered into images in the PDF in this MVP; we include chart counts/notes.
- next step: export plotly to static images (kaleido) and embed in report.

## Testing strategy

Pytest covers:

- schema inspector behavior
- code executor safety (blocked calls, allowed import stripping)
- agent workflow with LLM calls mocked (so CI doesn’t require Ollama)

CI:

- GitHub Actions runs `pytest` on push/PR.

## Demo script you can describe

1. Upload `sales_data.csv`
2. Auto-dashboard shows distribution + correlation + top values
3. Ask “Top 5 products by revenue” → table + bar chart
4. Ask “Monthly revenue trend” → line chart
5. Ask “Find outliers” → deterministic outlier detection + box plot
6. Export PDF

## Common issues & quick fixes

- **Port already in use**: `lsof -i :8000` then `kill <PID>`, or change ports.
- **403 on Streamlit uploads**: open `http://localhost:8501`; Streamlit config pins localhost and disables XSRF for local dev.
- **CI failing with Ollama connection refused**: mock LLM calls in tests (already done).

## Strong “what I’d do next” improvements

- Persist sessions (Redis/Postgres) and store uploaded data durably
- Hard sandbox code execution (subprocess + seccomp/container)
- Better chart selection (intent classifier / tool calling)
- Better PDF export (embed plot images)
- Add DB connectors (Postgres/MySQL) with SQL schema inspector

