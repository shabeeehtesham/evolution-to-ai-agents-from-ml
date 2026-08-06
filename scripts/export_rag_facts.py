# =====================================================================
# Export Project 4's real facts database for the live browser demo
# Author: Shabee Ibn Ehtesham
#
# Dumps SHAKESPEARE_FACTS_DB straight from rag_agent.py (loaded directly,
# so there's no copy-pasted, driftable duplicate) to a small JSON file
# that portfolio-website/app.js loads and runs its own from-scratch TF-IDF
# vector database over, in the browser - the same retrieve() algorithm
# SimpleVectorDB runs in Python, on the exact same facts.
# =====================================================================

import importlib.util
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULE_PATH = os.path.join(REPO_ROOT, "projects", "4_rag_agent", "rag_agent.py")
OUTPUT_PATH = os.path.join(REPO_ROOT, "portfolio-website", "model_weights", "rag_facts.json")


def load_rag_module():
    # Loaded by file path (folder name starts with a digit, not a valid
    # package name). A module name other than "__main__" means the script's
    # `if __name__ == "__main__":` CLI block never runs during this import.
    spec = importlib.util.spec_from_file_location("rag_agent_scratch", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    rag = load_rag_module()

    payload = {
        "documents": rag.SHAKESPEARE_FACTS_DB,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    print(f"[OK] Exported {len(payload['documents'])} facts to {OUTPUT_PATH} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
