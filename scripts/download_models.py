from __future__ import annotations

import shutil
from pathlib import Path
from huggingface_hub import snapshot_download

CLASSIFIER_REPO = "Bllossom/llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M"
ANSWER_REPO = "MLP-KTLim/llama-3-Korean-Bllossom-8B-gguf-Q4_K_M"

MODELS_DIR = Path("models")
HF_TMP_DIR = MODELS_DIR / "_hf"

CLASSIFIER_DST = MODELS_DIR / "llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M.gguf"
ANSWER_DST = MODELS_DIR / "llama-3-Korean-Bllossom-8B-gguf-Q4_K_M.gguf"

def _download_one(repo_id: str, dst_path: Path) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    local_repo_dir = Path(
        snapshot_download(
            repo_id=repo_id,
            local_dir=HF_TMP_DIR / repo_id.replace("/", "__"),
            local_dir_use_symlinks=False,
            allow_patterns=["*.gguf"],
        )
    )

    ggufs = list(local_repo_dir.rglob("*.gguf"))
    if not ggufs:
        raise RuntimeError(f"No .gguf found in repo: {repo_id}")

    gguf = max(ggufs, key=lambda p: p.stat().st_size)

    shutil.copy2(gguf, dst_path)

def main() -> None:
    try:
        print("[1/2] Download classifier GGUF from HF...")
        _download_one(CLASSIFIER_REPO, CLASSIFIER_DST)
        print(f"Saved: {CLASSIFIER_DST}")

        print("[2/2] Download answer GGUF from HF...")
        _download_one(ANSWER_REPO, ANSWER_DST)
        print(f"Saved: {ANSWER_DST}")

        print("Done. GGUF files are ready in ./models")
    finally:
        # 임시폴더 삭제
        shutil.rmtree(HF_TMP_DIR, ignore_errors=True)

if __name__ == "__main__":
    main()