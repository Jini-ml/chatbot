# 🧾 Tax AI Chatbot (LangChain + Ollama, GGUF Local)

세무 관련 질문만 답변하는 개인 프로젝트용 AI 챗봇입니다.

- GitHub에는 **코드만** 포함됩니다.
- 모델(GGUF)은 **Hugging Face에서 자동 다운로드(옵션 A)** 또는 **수동 다운로드(옵션 B)** 후 로컬에서 Ollama로 생성합니다.

## 모델 출처 (Hugging Face)
- 분류기(3B): https://huggingface.co/Bllossom/llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M
- 답변기(8B): https://huggingface.co/MLP-KTLim/llama-3-Korean-Bllossom-8B-gguf-Q4_K_M

> 모델 라이선스/사용 조건은 각 페이지를 따릅니다.

---

## 준비물
- Python 3.10+
- Ollama 설치 및 실행 중 (https://ollama.com/download)
- (권장) 8B 모델은 RAM 여유 필요

## 옵션 A : Hugging Face에서 자동 다운로드
    python scripts/download_models.py

    완료 후 아래 파일이 생성됩니다.

    models/
    ├── llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M.gguf
    └── llama-3-Korean-Bllossom-8B-gguf-Q4_K_M.gguf

## 옵션 B: 수동 다운로드

    아래 페이지에서 GGUF를 직접 다운로드하여 models/ 폴더에 아래 파일명으로 저장하세요.

    https://huggingface.co/Bllossom/llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M

    → models/llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M.gguf

    https://huggingface.co/MLP-KTLim/llama-3-Korean-Bllossom-8B-gguf-Q4_K_M

    → models/llama-3-Korean-Bllossom-8B-gguf-Q4_K_M.gguf

## Ollama 모델 생성
    python scripts/create_ollama_models.py

    성공하면 ollama list에 아래가 보입니다.

    bllossom-3b-classifier-q4km

    bllossom-8b-tax-answer-q4km