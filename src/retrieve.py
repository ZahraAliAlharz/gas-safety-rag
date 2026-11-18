"""Multilingual retrieval over the chunked corpus using intfloat/multilingual-e5-small
embeddings and cosine similarity. Passage embeddings are cached to disk so the
model only needs to run once per corpus version.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.ingest import Chunk, load_corpus

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
EMBEDDINGS_PATH = EMBEDDINGS_DIR / "chunk_embeddings.npy"
EMBEDDINGS_IDS_PATH = EMBEDDINGS_DIR / "chunk_ids.npy"

MODEL_NAME = "intfloat/multilingual-e5-small"
TOP_K = 3


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def embed_passages(chunks: list[Chunk]) -> np.ndarray:
    model = _get_model()
    texts = [f"passage: {c.text}" for c in chunks]
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return _normalize(vectors)


def embed_query(question: str) -> np.ndarray:
    model = _get_model()
    vector = model.encode([f"query: {question}"], convert_to_numpy=True, show_progress_bar=False)
    return _normalize(vector)[0]


def build_and_cache_embeddings(
    chunks: list[Chunk] | None = None,
    embeddings_path: Path = EMBEDDINGS_PATH,
    ids_path: Path = EMBEDDINGS_IDS_PATH,
) -> tuple[np.ndarray, list[str]]:
    if chunks is None:
        chunks = load_corpus()
    vectors = embed_passages(chunks)
    ids = [c.chunk_id for c in chunks]

    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, vectors)
    np.save(ids_path, np.array(ids))
    return vectors, ids


def load_or_build_embeddings(
    chunks: list[Chunk] | None = None,
    embeddings_path: Path = EMBEDDINGS_PATH,
    ids_path: Path = EMBEDDINGS_IDS_PATH,
) -> tuple[np.ndarray, list[str]]:
    if chunks is None:
        chunks = load_corpus()

    if embeddings_path.exists() and ids_path.exists():
        vectors = np.load(embeddings_path)
        ids = list(np.load(ids_path))
        expected_ids = [c.chunk_id for c in chunks]
        if ids == expected_ids:
            return vectors, ids
        # Corpus changed since the cache was built -- rebuild.

    return build_and_cache_embeddings(chunks, embeddings_path, ids_path)


class Retriever:
    def __init__(
        self,
        chunks: list[Chunk] | None = None,
        top_k: int = TOP_K,
        embeddings_path: Path = EMBEDDINGS_PATH,
        ids_path: Path = EMBEDDINGS_IDS_PATH,
    ):
        self.chunks = chunks if chunks is not None else load_corpus()
        self.chunks_by_id = {c.chunk_id: c for c in self.chunks}
        self.top_k = top_k
        self.vectors, self.ids = load_or_build_embeddings(self.chunks, embeddings_path, ids_path)

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        k = top_k if top_k is not None else self.top_k
        query_vec = embed_query(question)
        scores = self.vectors @ query_vec
        top_indices = np.argsort(-scores)[:k]
        return [
            RetrievedChunk(chunk=self.chunks_by_id[self.ids[i]], score=float(scores[i]))
            for i in top_indices
        ]


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    retriever = Retriever()
    for q in [
        "What should I do if liquid nitrogen contacts my skin?",
        "ما هي مخاطر الهيدروجين الغازي؟",
    ]:
        print(f"\nQ: {q}")
        for r in retriever.retrieve(q):
            print(f"  [{r.score:.3f}] {r.chunk.document} p{r.chunk.page}: {r.chunk.text[:80]!r}")
