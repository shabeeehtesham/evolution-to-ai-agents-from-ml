---
title: Evolution to AI Agents
emoji: 🧠
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---

# Evolution to AI Agents — server-side webapp

A FastAPI backend (real Python inference) + Streamlit frontend for the four
from-scratch architectures in this repo (MLP → RNN/LSTM → Transformer → RAG).
Every endpoint reconstructs the actual trained model objects from the weight
files already committed under `portfolio-website/model_weights/` and calls
each project's own forward-pass code directly — no reimplementation, no
mocked responses.

## Run locally without Docker

From the repo root:

```bash
pip install -r webapp/requirements.txt
cd webapp
uvicorn backend.main:app --reload --port 8000          # terminal 1
cd frontend && streamlit run Home.py                     # terminal 2
```

Streamlit defaults to `BACKEND_URL=http://localhost:8000` (override via env var).

## Run with Docker

From the repo root (the Dockerfile needs the monorepo root as build context,
since it also copies `projects/` and the exported weight files):

```bash
docker build -f webapp/Dockerfile -t evolution-ai-agents .
docker run -p 7860:7860 evolution-ai-agents
```

Open http://localhost:7860.

## Deploy to Hugging Face Spaces

1. Create a new Space, SDK = **Docker**.
2. Push this repo's contents to the Space's git remote (the Space's Dockerfile
   must resolve `projects/` and `portfolio-website/model_weights/` relative to
   the repo root exactly as this one does — if the Space repo *is* this whole
   monorepo, no changes are needed; if you only want `webapp/` in the Space,
   copy `projects/`, `portfolio-website/model_weights/`, and `webapp/` into it
   and move the Dockerfile to the Space root, updating the `COPY` paths).
3. The Space will build the image and serve it on port 7860 automatically.
