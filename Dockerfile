# AI Data Analyst - Backend + Frontend
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app

# Install system deps if needed (e.g. for reportlab)
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Default: run both via a script or expose ports for external processes
EXPOSE 8000 8501

# Run FastAPI by default; use docker-compose to run Streamlit as well
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
