# =====================================================================
# Export Project 3's trained Transformer weights for the live browser demo
# Author: Shabee Ibn Ehtesham
#
# Trains the actual PyTorch MiniTransformerGPT (same code path as
# mini_transformer_gpt.py, loaded directly so there's no duplicated
# training logic) and dumps its weights to a JSON file that
# portfolio-website/app.js loads and runs for real, in the browser,
# with a hand-written transformer forward pass (embeddings, causal
# self-attention per head, feedforward, layer norm) in plain JavaScript.
# No fake numbers - the visualizer runs the same pipeline the PyTorch
# model runs, using these exact trained parameters, including the real
# post-softmax attention weights for whatever the user types.
# =====================================================================

import importlib.util
import json
import os

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULE_PATH = os.path.join(REPO_ROOT, "projects", "3_transformer_scratch", "mini_transformer_gpt.py")
OUTPUT_PATH = os.path.join(REPO_ROOT, "portfolio-website", "model_weights", "transformer_weights.json")

# Matches the module's own --demo configuration exactly (already a proven,
# tested setup exercised by scripts/compare_text_generators.py) - keeps the
# exported weight file small and inference cheap enough to run per-keystroke
# in the browser, while still being the real architecture (multi-head causal
# self-attention, not a toy stand-in).
N_EMBED = 64
N_HEAD = 4
N_LAYER = 2
BLOCK_SIZE = 64
CORPUS_CHARS = 80_000
MAX_ITERS = 2000


def load_transformer_module():
    # Loaded by file path (folder name starts with a digit, not a valid
    # package name). A module name other than "__main__" means the script's
    # `if __name__ == "__main__":` CLI block never runs during this import.
    spec = importlib.util.spec_from_file_location("mini_transformer_scratch", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def round_nested(obj, decimals=6):
    if isinstance(obj, list):
        return [round_nested(x, decimals) for x in obj]
    return round(obj, decimals)


def main():
    tf = load_transformer_module()
    device = torch.device("cpu")

    text = tf.load_corpus(MODULE_PATH)[:CORPUS_CHARS]
    dataset = tf.CharDataset(text, block_size=BLOCK_SIZE)

    model = tf.MiniTransformerGPT(
        vocab_size=dataset.vocab_size, n_embed=N_EMBED, n_head=N_HEAD, n_layer=N_LAYER, block_size=BLOCK_SIZE
    ).to(device)
    model, final_loss = tf.train_transformer(model, dataset, device, max_iters=MAX_ITERS, batch_size=64, lr=1e-3)

    sd = model.state_dict()
    layers = []
    for l in range(N_LAYER):
        prefix = f"transformer.h.{l}."
        layers.append({
            "ln_1_weight": round_nested(sd[prefix + "ln_1.weight"].tolist()),
            "ln_1_bias": round_nested(sd[prefix + "ln_1.bias"].tolist()),
            "c_attn_weight": round_nested(sd[prefix + "attn.c_attn.weight"].tolist()),
            "c_attn_bias": round_nested(sd[prefix + "attn.c_attn.bias"].tolist()),
            "c_proj_weight": round_nested(sd[prefix + "attn.c_proj.weight"].tolist()),
            "c_proj_bias": round_nested(sd[prefix + "attn.c_proj.bias"].tolist()),
            "ln_2_weight": round_nested(sd[prefix + "ln_2.weight"].tolist()),
            "ln_2_bias": round_nested(sd[prefix + "ln_2.bias"].tolist()),
            "ffwd_w1": round_nested(sd[prefix + "ffwd.net.0.weight"].tolist()),
            "ffwd_b1": round_nested(sd[prefix + "ffwd.net.0.bias"].tolist()),
            "ffwd_w2": round_nested(sd[prefix + "ffwd.net.2.weight"].tolist()),
            "ffwd_b2": round_nested(sd[prefix + "ffwd.net.2.bias"].tolist()),
        })

    payload = {
        "vocab": dataset.chars,
        "n_embed": N_EMBED,
        "n_head": N_HEAD,
        "n_layer": N_LAYER,
        "block_size": BLOCK_SIZE,
        "wte": round_nested(sd["transformer.wte.weight"].tolist()),
        "wpe": round_nested(sd["transformer.wpe.weight"].tolist()),
        "ln_f_weight": round_nested(sd["transformer.ln_f.weight"].tolist()),
        "ln_f_bias": round_nested(sd["transformer.ln_f.bias"].tolist()),
        "lm_head_weight": round_nested(sd["lm_head.weight"].tolist()),
        "lm_head_bias": round_nested(sd["lm_head.bias"].tolist()),
        "layers": layers,
        "train_loss": round(float(final_loss), 4),
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    print(f"[OK] Exported trained Transformer weights to {OUTPUT_PATH} ({size_kb:.1f} KB)")
    print(f"     final train loss={payload['train_loss']}")


if __name__ == "__main__":
    main()
