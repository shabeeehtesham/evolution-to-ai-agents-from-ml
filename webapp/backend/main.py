# =====================================================================
# FastAPI backend for the "Evolution to AI Agents" server-side webapp.
#
# One endpoint family per architecture (MLP / RNN+LSTM / Transformer / RAG).
# Every endpoint calls real, already-trained model objects reconstructed by
# model_loader.py from the weight files under portfolio-website/model_weights/
# - this is genuine server-side Python inference, not a mock.
# =====================================================================

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import model_loader

app = FastAPI(title="Evolution to AI Agents - Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Static, once-measured perplexity/loss comparison across all four
# architectures on the identical task - see INTERVIEW_PREP.md. Deliberately
# not recomputed live (see that doc for why a live per-request metric would
# be a worse, more fragile design here).
ARCHITECTURE_COMPARISON = [
    {"project": 1, "arch": "MLP (fixed 10-char window)", "val_loss": 1.9896, "perplexity": 5.93},
    {"project": 2, "arch": "RNN (unbounded hidden state)", "val_loss": 1.7435, "perplexity": 4.72},
    {"project": 2, "arch": "LSTM (gated memory)", "val_loss": 1.7435, "perplexity": 4.40},
    {"project": 3, "arch": "Transformer (self-attention)", "val_loss": 1.2401, "perplexity": 3.45},
]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics/comparison")
def metrics_comparison():
    return {"architectures": ARCHITECTURE_COMPARISON}


# ---------------------------------------------------------------------
# Project 1: MLP
# ---------------------------------------------------------------------

class ScoreTextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)


class GenerateRequest(BaseModel):
    num_samples: int = Field(3, ge=1, le=10)
    temperature: float = Field(0.8, ge=0.1, le=2.0)
    max_len: int = Field(200, ge=20, le=500)


@app.post("/mlp/score")
def mlp_score(req: ScoreTextRequest):
    model = model_loader.get("mlp")
    return {"results": model.score_text(req.text)}


@app.post("/mlp/generate")
def mlp_generate(req: GenerateRequest):
    model = model_loader.get("mlp")
    samples = model.generate(num_samples=req.num_samples, temperature=req.temperature, max_len=req.max_len)
    return {"samples": samples, "val_loss": model.val_loss, "val_acc": model.val_acc}


# ---------------------------------------------------------------------
# Project 2: RNN & LSTM
# ---------------------------------------------------------------------

class SeqGenerateRequest(BaseModel):
    arch: str = Field(..., pattern="^(rnn|lstm)$")
    length: int = Field(200, ge=20, le=500)
    temperature: float = Field(0.8, ge=0.1, le=2.0)


class ReadAlongRequest(BaseModel):
    text: str = Field(..., min_length=2, max_length=2000)


@app.post("/rnn-lstm/generate")
def seq_generate(req: SeqGenerateRequest):
    model = model_loader.get(req.arch)
    text = model.generate(length=req.length, temperature=req.temperature)
    return {"text": text, "train_loss": model.train_loss}


@app.post("/rnn-lstm/read-along")
def seq_read_along(req: ReadAlongRequest):
    """Teacher-forced, zero-randomness comparison: both models read the exact
    same real text and report their own probability for the actual next
    character at every step. Architecture is the only variable."""
    rnn = model_loader.get("rnn")
    lstm = model_loader.get("lstm")
    rnn_conf = rnn.teacher_forced_confidence(req.text)
    lstm_conf = lstm.teacher_forced_confidence(req.text)
    return {
        "chars": list(req.text)[1: len(rnn_conf) + 1],
        "rnn_confidence": rnn_conf,
        "lstm_confidence": lstm_conf,
        "rnn_avg": sum(rnn_conf) / len(rnn_conf) if rnn_conf else 0.0,
        "lstm_avg": sum(lstm_conf) / len(lstm_conf) if lstm_conf else 0.0,
    }


# ---------------------------------------------------------------------
# Project 3: Transformer
# ---------------------------------------------------------------------

class TransformerGenerateRequest(BaseModel):
    prompt: str = Field("\n", max_length=64)
    max_new_tokens: int = Field(200, ge=20, le=500)
    temperature: float = Field(0.7, ge=0.1, le=2.0)
    top_k: int = Field(10, ge=1, le=66)


class AttentionRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=64)


@app.post("/transformer/generate")
def transformer_generate(req: TransformerGenerateRequest):
    model = model_loader.get("transformer")
    text = model.generate(prompt=req.prompt, max_new_tokens=req.max_new_tokens, temperature=req.temperature, top_k=req.top_k)
    return {"text": text, "train_loss": model.train_loss}


@app.post("/transformer/attention")
def transformer_attention(req: AttentionRequest):
    model = model_loader.get("transformer")
    return model.attention_for_prompt(req.prompt)


# ---------------------------------------------------------------------
# Project 4: RAG Agent
# ---------------------------------------------------------------------

class RagQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)


@app.post("/rag/query")
def rag_query(req: RagQueryRequest):
    model = model_loader.get("rag")
    return model.query(req.question)
