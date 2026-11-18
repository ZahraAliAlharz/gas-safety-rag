from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from src.retrieve import Retriever
from tests.fixtures import FIXTURE_CHUNKS

client = TestClient(app)


def _fixture_retriever(tmp_path: Path) -> Retriever:
    return Retriever(
        chunks=FIXTURE_CHUNKS,
        embeddings_path=tmp_path / "embeddings.npy",
        ids_path=tmp_path / "ids.npy",
    )


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@patch("src.answer.call_ollama")
def test_ask_no_rag_is_mocked_and_returns_shape(mock_call_ollama):
    mock_call_ollama.return_value = "Liquid nitrogen is extremely cold. [doc, page 1]"

    resp = client.post("/ask", json={"question": "What is liquid nitrogen?", "use_rag": False})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Liquid nitrogen is extremely cold. [doc, page 1]"
    assert body["retrieved_chunks"] == []
    assert body["citations"][0] == {"document": "doc", "page": 1}
    mock_call_ollama.assert_called_once()


@patch("src.answer.call_ollama")
@patch("app.main.get_retriever")
def test_ask_rag_mode_returns_retrieved_chunks(mock_get_retriever, mock_call_ollama, tmp_path):
    mock_get_retriever.return_value = _fixture_retriever(tmp_path)
    mock_call_ollama.return_value = "Not covered in the provided documents"

    resp = client.post(
        "/ask", json={"question": "What should I do if liquid nitrogen touches my skin?", "use_rag": True}
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Not covered in the provided documents"
    assert body["citations"] == []
    assert len(body["retrieved_chunks"]) == 3
    assert all("document" in c and "page" in c and "text" in c for c in body["retrieved_chunks"])


def test_ask_rejects_empty_question():
    resp = client.post("/ask", json={"question": "", "use_rag": False})
    assert resp.status_code == 422
