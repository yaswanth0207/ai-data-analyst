# AI Data Analyst Agent

A multi-step AI agent where users upload CSV files, ask questions in plain English, and get answers with code, visualizations, and insights. Built with FastAPI, Streamlit, LangGraph, and Ollama.

## Tech Stack

- **Backend:** FastAPI (Python)
- **Frontend:** Streamlit
- **Agent:** LangGraph
- **LLM:** Ollama (local) — e.g. `llama3.2` or `qwen2.5-coder:3b`
- **Data:** pandas, numpy; **Viz:** plotly; **Memory:** ChromaDB; **Export:** reportlab (PDF)

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) running locally with a model (e.g. `ollama pull llama3.2`)

## Setup

```bash
cd ai-data-analyst
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env      # edit if needed
```

## Run locally

1. Start Ollama (and pull a model if needed):
   ```bash
   ollama serve
   ollama pull llama3.2
   ```

2. Start the backend:
   ```bash
   PYTHONPATH=. uvicorn backend.main:app --reload --port 8000
   ```

3. In another terminal, start the frontend:
   ```bash
   streamlit run frontend/app.py --server.port 8501
   ```

4. Open http://localhost:8501, upload a CSV, click "Load & Analyze", then ask questions.

### Port already in use

If you see `[Errno 48] Address already in use` when starting the backend:

- **Free port 8000:**  
  `lsof -i :8000` to find the PID, then `kill <PID>`
- **Or use another port:**  
  `PYTHONPATH=. uvicorn backend.main:app --reload --port 8003`  
  Then set `API_BASE_URL=http://localhost:8003` when starting Streamlit (or in `.env`).

## Docker

```bash
docker-compose up --build
```

- Backend: http://localhost:8000  
- Frontend: http://localhost:8501  

Ensure Ollama is reachable (e.g. on host use `OLLAMA_BASE_URL=http://host.docker.internal:11434`).

## Demo flow

1. Upload a CSV (e.g. sales data).
2. Auto-dashboard appears with key charts and insights.
3. Ask: "What are the top 5 products by revenue?" → table + bar chart.
4. Ask: "Show me monthly sales trend" → line chart.
5. Ask: "Find any anomalies" → list of outliers.
6. Ask: "Summarize this dataset" → plain English summary.
7. Click "Download PDF Report" for a session report.

## Demo dataset

This repo includes a realistic sample dataset:

- `sales_data.csv` (≈200 rows)
- Mixed types: numeric (`revenue`, `units`), categorical (`product`, `region`, `channel`, `customer_type`), datetime-like (`order_date`, `month`), boolean (`returned`)

Example questions to try:

- "What are the top 5 products by revenue?"
- "Show me monthly revenue trend"
- "Which region has the highest average order revenue?"
- "What percentage of orders were returned?"
- "Are discounts correlated with revenue or units?"
- "Find anomalies in revenue (outliers)"

## Architecture

```mermaid
flowchart LR
  U[User] -->|CSV upload / questions| S[Streamlit UI]
  S -->|HTTP| API[FastAPI]
  API -->|load + schema| INS[Schema Inspector<br/>(pandas)]
  API -->|invoke| LG[LangGraph Agent]
  LG --> RT[Router]
  RT --> CG[Ollama Code Generator]
  CG --> EX[Sandbox Executor]
  EX -->|result/fig/error| ANS[Answer Node]
  API --> MEM[ChromaDB Query Memory]
  API --> PDF[ReportLab PDF Export]
  API --> S
```

## Project structure

```
ai-data-analyst/
├── backend/
│   ├── main.py              # FastAPI app
│   ├── agent/               # LangGraph state, graph, nodes, tools
│   ├── analyst/             # schema, code gen, executor, viz, insights
│   ├── memory/              # ChromaDB query memory
│   ├── export/              # PDF report
│   └── utils/               # file upload
├── frontend/
│   └── app.py               # Streamlit UI
├── requirements.txt
├── .env.example
├── Dockerfile
└── docker-compose.yml
```

## License

MIT
