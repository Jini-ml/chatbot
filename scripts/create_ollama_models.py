from __future__ import annotations

import subprocess
from pathlib import Path
import sys

CLASSIFIER_MODEL_NAME = "bllossom_3b_classifier:q4km"
ANSWER_MODEL_NAME = "bllossom_8b_tax_answer:q4km"

CLASSIFIER_GGUF = Path("models/llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M.gguf")
ANSWER_GGUF = Path("models/llama-3-Korean-Bllossom-8B-gguf-Q4_K_M.gguf")

def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)

def main() -> None:
    if not CLASSIFIER_GGUF.exists():
        print(f"Missing: {CLASSIFIER_GGUF}")
        print("Run: python scripts/download_models.py (or download manually)")
        sys.exit(1)

    if not ANSWER_GGUF.exists():
        print(f"Missing: {ANSWER_GGUF}")
        print("Run: python scripts/download_models.py (or download manually)")
        sys.exit(1)

    print("[1/2] Creating classifier model in Ollama...")
    run(["ollama", "create", CLASSIFIER_MODEL_NAME, "-f", "models/Modelfile.classifier"])

    print("[2/2] Creating answer model in Ollama...")
    run(["ollama", "create", ANSWER_MODEL_NAME, "-f", "models/Modelfile.answer"])

    print("Done. Check:")
    run(["ollama", "list"])

if __name__ == "__main__":
    main()