import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from chatbot_service import process_question


app = FastAPI(
    title="세무 AI 챗봇 API",
    version="1.0.0",
)


allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://localhost:8000",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=2000,
        description="사용자의 질문",
    )
    session_id: str = Field(
        min_length=1,
        max_length=100,
        description="사용자별 대화 세션 ID",
    )


class ChatResponse(BaseModel):
    label: str
    answer: str


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "세무 AI 챗봇 API",
        "docs": "/docs",
    }


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        result = process_question(
            question=request.question,
            base_session_id=request.session_id,
        )

        return ChatResponse(**result)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        # 운영 환경에서는 내부 오류 내용을 그대로 사용자에게 보내지 않습니다.
        print(f"Chatbot error: {exc!r}")

        raise HTTPException(
            status_code=503,
            detail="AI 모델 서버에 연결할 수 없거나 응답 처리 중 오류가 발생했습니다.",
        ) from exc