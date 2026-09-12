#!/bin/sh
set -e

# FastAPI backend, internal only (not exposed by the container).
uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for it to actually be ready before Streamlit starts hitting it.
for i in $(seq 1 30); do
  if python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=1)" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

# Streamlit frontend, the container's actual public entrypoint.
cd frontend
streamlit run Home.py --server.address 0.0.0.0 --server.port 7860 --server.headless true

kill "$BACKEND_PID" 2>/dev/null || true
