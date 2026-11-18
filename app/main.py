"""FastAPI service exposing the bilingual gas-safety RAG assistant."""
from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.answer import AnswerResult, answer_question
from src.retrieve import Retriever

app = FastAPI(title="Bilingual Gas Safety Assistant")


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever()


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    use_rag: bool = True


class RetrievedChunkOut(BaseModel):
    document: str
    page: int
    score: float
    text: str


class CitationOut(BaseModel):
    document: str
    page: int


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    retrieved_chunks: list[RetrievedChunkOut]


class HealthResponse(BaseModel):
    status: str


def _to_response(result: AnswerResult) -> AskResponse:
    return AskResponse(
        answer=result.answer,
        citations=[CitationOut(document=c.document, page=c.page) for c in result.citations],
        retrieved_chunks=[
            RetrievedChunkOut(
                document=r.chunk.document, page=r.chunk.page, score=r.score, text=r.chunk.text
            )
            for r in result.retrieved_chunks
        ],
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    try:
        result = answer_question(
            request.question,
            use_rag=request.use_rag,
            retriever=get_retriever() if request.use_rag else None,
        )
    except Exception as exc:  # noqa: BLE001 - surface LLM/retrieval failures as 502s
        raise HTTPException(status_code=502, detail=f"answering failed: {exc}") from exc
    return _to_response(result)
