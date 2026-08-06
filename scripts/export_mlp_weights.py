# =====================================================================
# Export Project 1's trained MLP weights for the live browser demo
# Author: Shabee Ibn Ehtesham
#
# Trains the actual from-scratch NumPy MLP (same code path as
# `neural_network_scratch.py`, loaded directly so there's no duplicated
# training logic) and dumps its weights to a small JSON file that
# portfolio-website/app.js loads and runs for real, in the browser, with
# plain JavaScript matrix math. No fake numbers - the visualizer performs
# the same one-hot -> Dense -> ReLU -> Dense -> Softmax pipeline the
# Python model runs, using these exact trained parameters.
# =====================================================================

import importlib.util
import json
import os

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULE_PATH = os.path.join(REPO_ROOT, "projects", "1_neural_network_scratch", "neural_network_scratch.py")
OUTPUT_PATH = os.path.join(REPO_ROOT, "portfolio-website", "model_weights", "mlp_weights.json")


def load_mlp_module():
    # Loaded by file path (not a dotted import) since the containing folder
    # name starts with a digit and isn't a valid Python package name. Using a
    # module name other than "__main__" means the script's `if __name__ ==
    # "__main__":` CLI block never runs during this import.
    spec = importlib.util.spec_from_file_location("mlp_scratch", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BLOCK_SIZE = 10
HIDDEN_SIZE = 128
CORPUS_CHARS = 200_000


def main():
    mlp = load_mlp_module()

    text = mlp.load_corpus(MODULE_PATH)[:CORPUS_CHARS]
    vocab, stoi, itos = mlp.build_vocab(text)
    X, Y = mlp.build_dataset(text, stoi, block_size=BLOCK_SIZE)
    indices = np.random.permutation(len(X))
    split = int(0.9 * len(X))
    X_train, X_test = X[indices[:split]], X[indices[split:]]
    y_train, y_test = Y[indices[:split]], Y[indices[split:]]

    layer1, activation1, layer2, _, _, test_loss, test_acc = mlp.run_training(
        X_train, y_train, X_test, y_test,
        hidden_size=HIDDEN_SIZE, num_classes=len(vocab), epochs=18, batch_size=256, lr=0.1, decay=1e-3, momentum=0.9
    )

    payload = {
        "vocab": vocab,
        "block_size": BLOCK_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "w1": np.round(layer1.weights, 6).tolist(),
        "b1": np.round(layer1.biases, 6).tolist(),
        "w2": np.round(layer2.weights, 6).tolist(),
        "b2": np.round(layer2.biases, 6).tolist(),
        "val_loss": round(float(test_loss), 4),
        "val_acc": round(float(test_acc), 4),
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    print(f"[OK] Exported trained MLP weights to {OUTPUT_PATH} ({size_kb:.1f} KB)")
    print(f"     val_loss={payload['val_loss']}  val_acc={payload['val_acc']}")


if __name__ == "__main__":
    main()
