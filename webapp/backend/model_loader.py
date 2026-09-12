# =====================================================================
# Real-inference model loader for the server-side webapp.
#
# Every project folder under projects/ is a valid module by file path (not
# a dotted import, since "1_neural_network_scratch" etc. aren't valid
# Python package names - same pattern already used by scripts/export_*.py).
# Rather than re-implement any forward pass, this file reconstructs the
# *actual* trained model objects (numpy Layer_Dense, torch CharRNN/CharLSTM,
# torch MiniTransformerGPT) from the weight files already committed under
# portfolio-website/model_weights/, then calls each project's own
# generate_sentence()/forward() functions directly. No duplicated math.
# =====================================================================

import importlib.util
import json
import os
import threading
import types

import numpy as np
import torch

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
WEBAPP_DIR = os.path.dirname(BACKEND_DIR)
REPO_ROOT = os.path.dirname(WEBAPP_DIR)
WEIGHTS_DIR = os.path.join(REPO_ROOT, "portfolio-website", "model_weights")

_load_lock = threading.Lock()
_singletons = {}


def _import_module(name, rel_path):
    path = os.path.join(REPO_ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_weights(filename):
    with open(os.path.join(WEIGHTS_DIR, filename), encoding="utf-8") as f:
        return json.load(f)


class MLPModel:
    """Fixed-10-char-window MLP (Project 1), real forward pass via the
    project's own Layer_Dense/Activation_ReLU classes."""

    def __init__(self):
        self.mod = _import_module("mlp_scratch_backend", "projects/1_neural_network_scratch/neural_network_scratch.py")
        w = _load_weights("mlp_weights.json")

        self.vocab = w["vocab"]
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}
        self.block_size = w["block_size"]
        self.pad_idx = self.stoi.get("\x00", 0)

        self.layer1 = self.mod.Layer_Dense(len(self.vocab) * self.block_size, w["hidden_size"])
        self.layer1.weights = np.array(w["w1"], dtype=np.float64)
        self.layer1.biases = np.array(w["b1"], dtype=np.float64)
        self.activation1 = self.mod.Activation_ReLU()
        self.layer2 = self.mod.Layer_Dense(w["hidden_size"], len(self.vocab))
        self.layer2.weights = np.array(w["w2"], dtype=np.float64)
        self.layer2.biases = np.array(w["b2"], dtype=np.float64)

        self.val_loss = w["val_loss"]
        self.val_acc = w["val_acc"]

    def _forward(self, x):
        self.layer1.forward(x)
        self.activation1.forward(self.layer1.output)
        self.layer2.forward(self.activation1.output)
        return self.layer2.output[0]

    def score_text(self, text, top_k=5):
        """For each character, run the real forward pass on the block_size
        characters before it and report where the true next character
        ranked among the model's predictions - the server-side twin of the
        browser demo's mlpScoreText()."""
        results = []
        context = [self.pad_idx] * self.block_size
        for ch in text:
            x = self.mod.encode_context(context, len(self.vocab)).reshape(1, -1)
            logits = self._forward(x)
            exp = np.exp(logits - np.max(logits))
            probs = exp / exp.sum()
            order = np.argsort(-probs)
            true_idx = self.stoi.get(ch, self.pad_idx)
            rank = int(np.where(order == true_idx)[0][0]) + 1
            top = [{"char": self.itos[int(i)], "prob": float(probs[i])} for i in order[:top_k]]
            results.append({"char": ch, "rank": rank, "prob": float(probs[true_idx]), "top": top})
            context = context[1:] + [true_idx]
        return results

    def generate(self, num_samples=3, temperature=0.8, max_len=200):
        return self.mod.generate_sentences(
            self.layer1, self.activation1, self.layer2, self.stoi, self.itos,
            num_sentences=num_samples, block_size=self.block_size, temperature=temperature,
        )


class SeqModel:
    """Project 2's CharRNN or CharLSTM, real forward pass via the project's
    own generate_text() function, weights loaded from the exported JSON."""

    def __init__(self, arch):
        assert arch in ("rnn", "lstm")
        self.arch = arch
        self.mod = _import_module(f"char_rnn_backend_{arch}", "projects/2_rnn_lstm_generator/char_rnn_generator.py")
        w = _load_weights("rnn_weights.json" if arch == "rnn" else "lstm_weights.json")

        self.vocab = w["vocab"]
        self.char2idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx2char = {i: c for i, c in enumerate(self.vocab)}
        self.train_loss = w["train_loss"]
        self.hidden_size = w["hidden_size"]

        cls = self.mod.CharRNN if arch == "rnn" else self.mod.CharLSTM
        model = cls(len(self.vocab), embed_size=w["embed_size"], hidden_size=w["hidden_size"], n_layers=1)
        sd = model.state_dict()
        rnn_key = "rnn" if arch == "rnn" else "lstm"
        sd["encoder.weight"] = torch.tensor(w["embedding_weight"], dtype=torch.float32)
        sd[f"{rnn_key}.weight_ih_l0"] = torch.tensor(w["w_ih"], dtype=torch.float32)
        sd[f"{rnn_key}.weight_hh_l0"] = torch.tensor(w["w_hh"], dtype=torch.float32)
        sd[f"{rnn_key}.bias_ih_l0"] = torch.tensor(w["b_ih"], dtype=torch.float32)
        sd[f"{rnn_key}.bias_hh_l0"] = torch.tensor(w["b_hh"], dtype=torch.float32)
        sd["decoder.weight"] = torch.tensor(w["decoder_weight"], dtype=torch.float32)
        sd["decoder.bias"] = torch.tensor(w["decoder_bias"], dtype=torch.float32)
        model.load_state_dict(sd)
        model.eval()
        self.model = model
        self.device = torch.device("cpu")

    def _dataset_stub(self):
        return types.SimpleNamespace(char2idx=self.char2idx, idx2char=self.idx2char)

    def generate(self, length=200, temperature=0.8):
        return self.mod.generate_text(self.model, self._dataset_stub(), self.device, predict_len=length, temperature=temperature)

    @torch.no_grad()
    def teacher_forced_confidence(self, text):
        """Feeds real text one character at a time with zero randomness and
        records the model's own probability for the *actual* next character
        at each step - the numeric signal behind the RNN-vs-LSTM read-along."""
        clean = [c for c in text if c in self.char2idx]
        if len(clean) < 2:
            return []
        hidden = self.model.init_hidden(1, self.device)
        confidences = []
        for i in range(len(clean) - 1):
            x = torch.tensor([[self.char2idx[clean[i]]]], dtype=torch.long)
            logits, hidden = self.model(x, hidden)
            probs = torch.softmax(logits[-1], dim=-1)
            confidences.append(float(probs[self.char2idx[clean[i + 1]]]))
        return confidences


class TransformerModel:
    """Project 3's decoder-only Mini-GPT, real forward pass reconstructed
    from its own already-loaded, weight-loaded submodules (c_attn/c_proj/
    layer norms) - see forward_capture_attention() below."""

    def __init__(self):
        self.mod = _import_module("mini_transformer_backend", "projects/3_transformer_scratch/mini_transformer_gpt.py")
        w = _load_weights("transformer_weights.json")

        self.vocab = w["vocab"]
        self.char2idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx2char = {i: c for i, c in enumerate(self.vocab)}
        self.n_head = w["n_head"]
        self.n_layer = w["n_layer"]
        self.block_size = w["block_size"]
        self.train_loss = w["train_loss"]

        model = self.mod.MiniTransformerGPT(
            vocab_size=len(self.vocab), n_embed=w["n_embed"], n_head=w["n_head"],
            n_layer=w["n_layer"], block_size=w["block_size"],
        )
        sd = model.state_dict()
        sd["transformer.wte.weight"] = torch.tensor(w["wte"], dtype=torch.float32)
        sd["transformer.wpe.weight"] = torch.tensor(w["wpe"], dtype=torch.float32)
        sd["transformer.ln_f.weight"] = torch.tensor(w["ln_f_weight"], dtype=torch.float32)
        sd["transformer.ln_f.bias"] = torch.tensor(w["ln_f_bias"], dtype=torch.float32)
        sd["lm_head.weight"] = torch.tensor(w["lm_head_weight"], dtype=torch.float32)
        sd["lm_head.bias"] = torch.tensor(w["lm_head_bias"], dtype=torch.float32)
        for l, layer in enumerate(w["layers"]):
            p = f"transformer.h.{l}."
            sd[p + "ln_1.weight"] = torch.tensor(layer["ln_1_weight"], dtype=torch.float32)
            sd[p + "ln_1.bias"] = torch.tensor(layer["ln_1_bias"], dtype=torch.float32)
            sd[p + "attn.c_attn.weight"] = torch.tensor(layer["c_attn_weight"], dtype=torch.float32)
            sd[p + "attn.c_attn.bias"] = torch.tensor(layer["c_attn_bias"], dtype=torch.float32)
            sd[p + "attn.c_proj.weight"] = torch.tensor(layer["c_proj_weight"], dtype=torch.float32)
            sd[p + "attn.c_proj.bias"] = torch.tensor(layer["c_proj_bias"], dtype=torch.float32)
            sd[p + "ln_2.weight"] = torch.tensor(layer["ln_2_weight"], dtype=torch.float32)
            sd[p + "ln_2.bias"] = torch.tensor(layer["ln_2_bias"], dtype=torch.float32)
            sd[p + "ffwd.net.0.weight"] = torch.tensor(layer["ffwd_w1"], dtype=torch.float32)
            sd[p + "ffwd.net.0.bias"] = torch.tensor(layer["ffwd_b1"], dtype=torch.float32)
            sd[p + "ffwd.net.2.weight"] = torch.tensor(layer["ffwd_w2"], dtype=torch.float32)
            sd[p + "ffwd.net.2.bias"] = torch.tensor(layer["ffwd_b2"], dtype=torch.float32)
        model.load_state_dict(sd)
        model.eval()
        self.model = model
        self.device = torch.device("cpu")

    def _dataset_stub(self):
        return types.SimpleNamespace(char2idx=self.char2idx, idx2char=self.idx2char)

    def generate(self, prompt="\n", max_new_tokens=200, temperature=0.7, top_k=10):
        model, mod = self.model, self.mod
        model.eval()
        stop_tokens = {self.char2idx[c] for c in mod.STOP_CHARS if c in self.char2idx}
        prompt = prompt if prompt else "\n"
        ids = [self.char2idx.get(c, self.char2idx.get("\n", 0)) for c in prompt]
        context = torch.tensor([ids], dtype=torch.long)
        gen_ids = model.generate(context, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, stop_tokens=stop_tokens)[0].tolist()
        return "".join(self.idx2char[i] for i in gen_ids)

    @torch.no_grad()
    def attention_for_prompt(self, prompt):
        """Real post-softmax causal self-attention weights for `prompt`,
        computed by calling this model's own trained c_attn/c_proj/layer-norm
        submodules directly (not a reimplementation, just exposing the
        intermediate `att` tensor that CausalSelfAttention.forward computes
        internally but doesn't return)."""
        prompt = prompt[-self.block_size:] if len(prompt) > self.block_size else prompt
        prompt = prompt or "\n"
        ids = [self.char2idx.get(c, self.char2idx.get("\n", 0)) for c in prompt]
        idx = torch.tensor([ids], dtype=torch.long)
        model = self.model

        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long).unsqueeze(0)
        x = model.transformer.wte(idx) + model.transformer.wpe(pos)

        attentions = []
        for block in model.transformer.h:
            ln1 = block.ln_1(x)
            attn = block.attn
            B, T, C = ln1.size()
            q, k, v = attn.c_attn(ln1).split(attn.n_embed, dim=2)
            k = k.view(B, T, attn.n_head, C // attn.n_head).transpose(1, 2)
            q = q.view(B, T, attn.n_head, C // attn.n_head).transpose(1, 2)
            v = v.view(B, T, attn.n_head, C // attn.n_head).transpose(1, 2)
            att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))
            att = att.masked_fill(attn.bias[:, :, :T, :T] == 0, float("-inf"))
            att = torch.softmax(att, dim=-1)
            attentions.append(att[0].detach().tolist())  # (n_head, T, T)
            y = att @ v
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            y = attn.c_proj(y)
            x = x + y
            x = x + block.ffwd(block.ln_2(x))

        return {"tokens": list(prompt), "attentions": attentions}


class RagModel:
    """Project 4's RAG agent, used exactly as-is - no reconstruction needed,
    since it never required training (TF-IDF is built in-process)."""

    def __init__(self):
        self.mod = _import_module("rag_agent_backend", "projects/4_rag_agent/rag_agent.py")
        self.agent = self.mod.RAGAgent(self.mod.SHAKESPEARE_FACTS_DB)

    def query(self, question):
        retrieved = self.agent.db.retrieve(question, top_k=2)
        answer = self.agent.query(question)
        return {
            "retrieved": [{"score": r["score"], "text": r["document"]} for r in retrieved],
            "answer": answer,
        }


def get(name):
    """Lazily builds and caches each model singleton on first request, so
    server startup is instant and only the models actually used pay the
    (sub-second - these are tiny) reconstruction cost."""
    with _load_lock:
        if name not in _singletons:
            if name == "mlp":
                _singletons[name] = MLPModel()
            elif name in ("rnn", "lstm"):
                _singletons[name] = SeqModel(name)
            elif name == "transformer":
                _singletons[name] = TransformerModel()
            elif name == "rag":
                _singletons[name] = RagModel()
            else:
                raise KeyError(f"Unknown model: {name}")
        return _singletons[name]
