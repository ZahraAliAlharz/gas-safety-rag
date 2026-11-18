from pathlib import Path

from src.retrieve import Retriever
from tests.fixtures import FIXTURE_CHUNKS


def _make_retriever(tmp_path: Path) -> Retriever:
    return Retriever(
        chunks=FIXTURE_CHUNKS,
        embeddings_path=tmp_path / "embeddings.npy",
        ids_path=tmp_path / "ids.npy",
    )


def test_retrieve_returns_top_k_in_score_order(tmp_path):
    retriever = _make_retriever(tmp_path)
    results = retriever.retrieve("What is the boiling point of liquid nitrogen?", top_k=3)
    assert len(results) == 3
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_finds_correct_document_for_known_query(tmp_path):
    retriever = _make_retriever(tmp_path)
    results = retriever.retrieve("What PPE should I wear when handling liquid oxygen?")
    documents = [r.chunk.document for r in results]
    assert "safetygram-6-liquid-oxygen" in documents


def test_retrieve_works_for_arabic_query_same_topic(tmp_path):
    retriever = _make_retriever(tmp_path)
    results = retriever.retrieve("ما هي مخاطر استنشاق الهيدروجين الغازي؟")
    documents = [r.chunk.document for r in results]
    # Either hydrogen Safetygram is an acceptable hit for a hydrogen-inhalation query.
    assert any("hydrogen" in d for d in documents)
