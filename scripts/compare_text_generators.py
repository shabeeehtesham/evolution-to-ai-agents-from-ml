# =====================================================================
# Cross-Project Comparison: Same Task, Better Tool
# Author: Shabee Ibn Ehtesham
#
# Runs Projects 1-3's fast-but-real "--demo" training mode concurrently and
# renders a side-by-side comparison of generated Shakespeare-style text and
# training loss, making the MLP -> RNN/LSTM -> Transformer quality
# progression visible in one place. Project 4 (RAG) isn't included here - it
# answers grounded factual questions rather than generating text, so it
# isn't directly comparable on this axis.
#
# `projects/1_neural_network_scratch` starts with a digit, so none of these
# scripts can be dotted-imported as Python modules - subprocesses sidestep
# that entirely and reuse each project's existing CLI/argparse plumbing.
# =====================================================================

import subprocess
import sys
import os
import json
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SENTINEL = "###RESULT_JSON### "
# Larger context windows (block_size/seq_len for sentence-level text, vs. the
# short fixed windows used for name generation) cost meaningfully more per
# step, especially the Transformer's O(block_size^2) attention - measured
# demo runs take up to ~3 minutes for the Transformer alone, so this timeout
# has real headroom above that rather than being tight.
TIMEOUT_SECONDS = 480

PROJECTS = [
    {
        "label": "1. MLP (fixed 10-char window)",
        "script": os.path.join(REPO_ROOT, "projects", "1_neural_network_scratch", "neural_network_scratch.py"),
        "args": ["--demo"],
    },
    {
        "label": "2. LSTM (sequential hidden state)",
        "script": os.path.join(REPO_ROOT, "projects", "2_rnn_lstm_generator", "char_rnn_generator.py"),
        "args": ["--demo", "--model", "lstm"],
    },
    {
        "label": "3. Transformer (full self-attention)",
        "script": os.path.join(REPO_ROOT, "projects", "3_transformer_scratch", "mini_transformer_gpt.py"),
        "args": ["--demo"],
    },
]


def launch(project):
    cmd = [sys.executable, project["script"]] + project["args"]
    return subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", cwd=REPO_ROOT
    )


def parse_result(project, stdout, stderr, returncode):
    if returncode != 0:
        print(f"[FAIL] {project['label']} exited with code {returncode}.")
        print(stderr.strip()[-2000:])
        return None

    sentinel_line = None
    for line in stdout.splitlines():
        if line.startswith(SENTINEL):
            sentinel_line = line[len(SENTINEL):]

    if sentinel_line is None:
        print(f"[FAIL] {project['label']} produced no ###RESULT_JSON### line.")
        print(stdout.strip()[-2000:])
        return None

    try:
        return json.loads(sentinel_line)
    except json.JSONDecodeError as e:
        print(f"[FAIL] {project['label']} produced malformed JSON ({e}).")
        return None


def main():
    print("=== Launching Projects 1-3 concurrently (--demo mode) ===")
    start = time.time()

    running = []
    for project in PROJECTS:
        print(f"  Starting: {project['label']}")
        running.append((project, launch(project)))

    results = []
    for project, proc in running:
        try:
            stdout, stderr = proc.communicate(timeout=TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            print(f"[FAIL] {project['label']} timed out after {TIMEOUT_SECONDS}s.")
            results.append(None)
            continue
        results.append(parse_result(project, stdout, stderr, proc.returncode))

    elapsed = time.time() - start
    print(f"\nAll subprocesses finished in {elapsed:.1f}s (wall time).\n")

    failed_labels = [p["label"] for (p, _), r in zip(running, results) if r is None]
    if failed_labels:
        print(f"[ERROR] {len(failed_labels)} project(s) failed to produce results: {', '.join(failed_labels)}")
        sys.exit(1)

    print("=" * 78)
    print("SAME TASK, BETTER TOOL: character-level Shakespeare text generation across 3 architectures")
    print("=" * 78)
    print("Note: a few minutes of demo-scale training will not produce fully fluent")
    print("sentences from any of these - expect improving loss numbers and improving")
    print("local structure (real word fragments), not polished prose. The loss")
    print("numbers are the real evidence here; samples are shown as honest, short,")
    print("imperfect demo output.")

    for result in results:
        print(f"\n--- {result['arch']} ---")
        acc_str = f" | Validation accuracy: {result['val_acc']}" if "val_acc" in result else ""
        print(f"Loss:          {result['val_loss']}{acc_str}")
        print(f"Training time: {result['train_seconds']}s")
        print("Sample generated text:")
        for sample in result["samples"]:
            print(f"  - {sample!r}")

    print("\n" + "=" * 78)
    print("Loss (lower = better) - should decrease as context handling improves:")
    for result in results:
        print(f"  {result['arch']:<40} loss = {result['val_loss']}")
    print("=" * 78)


if __name__ == "__main__":
    main()
