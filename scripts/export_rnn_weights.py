# =====================================================================
# Export Project 2's trained plain-RNN weights for the live browser demo
# Author: Shabee Ibn Ehtesham
#
# Sibling to export_lstm_weights.py: trains the actual PyTorch CharRNN
# (same code path as char_rnn_generator.py, loaded directly) and dumps
# its weights to a small JSON file. This exists specifically so the
# browser demo can toggle between RNN and LSTM on the *same* typed text
# and show the vanishing-gradient problem live - the RNN's confidence on
# long-range dependencies should degrade faster than the LSTM's.
# =====================================================================

import importlib.util
import json
import os

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULE_PATH = os.path.join(REPO_ROOT, "projects", "2_rnn_lstm_generator", "char_rnn_generator.py")
OUTPUT_PATH = os.path.join(REPO_ROOT, "portfolio-website", "model_weights", "rnn_weights.json")

EMBED_SIZE = 32
HIDDEN_SIZE = 128
SEQ_LEN = 128
CORPUS_CHARS = 300_000


def load_rnn_module():
    spec = importlib.util.spec_from_file_location("char_rnn_scratch", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def round_nested(obj, decimals=6):
    if isinstance(obj, list):
        return [round_nested(x, decimals) for x in obj]
    return round(obj, decimals)


def main():
    rnn = load_rnn_module()
    device = torch.device("cpu")

    text = rnn.load_corpus(MODULE_PATH)[:CORPUS_CHARS]
    dataset = rnn.TextDataset(text, seq_len=SEQ_LEN)

    model = rnn.CharRNN(dataset.vocab_size, embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE, n_layers=1).to(device)
    model, final_loss = rnn.train_model(
        model, dataset, device, epochs=15, batch_size=64, lr=0.005, seq_len=SEQ_LEN, model_name="RNN"
    )

    sd = model.state_dict()
    payload = {
        "vocab": dataset.chars,
        "embed_size": EMBED_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "embedding_weight": round_nested(sd["encoder.weight"].tolist()),
        "w_ih": round_nested(sd["rnn.weight_ih_l0"].tolist()),
        "w_hh": round_nested(sd["rnn.weight_hh_l0"].tolist()),
        "b_ih": round_nested(sd["rnn.bias_ih_l0"].tolist()),
        "b_hh": round_nested(sd["rnn.bias_hh_l0"].tolist()),
        "decoder_weight": round_nested(sd["decoder.weight"].tolist()),
        "decoder_bias": round_nested(sd["decoder.bias"].tolist()),
        "train_loss": round(float(final_loss), 4),
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    print(f"[OK] Exported trained RNN weights to {OUTPUT_PATH} ({size_kb:.1f} KB)")
    print(f"     final train loss={payload['train_loss']}")


if __name__ == "__main__":
    main()
