"""Extract text from the Safetygram PDFs and chunk it into overlapping,
token-sized pieces with (document, page) metadata attached to each chunk.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import tiktoken
from pypdf import PdfReader

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PDF_DIR = DATA_DIR / "pdfs"
CHUNKS_PATH = DATA_DIR / "chunks.json"

CHUNK_TOKENS = 400
OVERLAP_TOKENS = 80

_ENCODING = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    chunk_id: str
    document: str
    page: int
    text: str


def extract_pages(pdf_path: Path) -> list[str]:
    """Return a list of page texts (1 entry per page, in page order)."""
    reader = PdfReader(str(pdf_path))
    return [page.extract_text() or "" for page in reader.pages]


def chunk_page_text(text: str, chunk_tokens: int = CHUNK_TOKENS, overlap_tokens: int = OVERLAP_TOKENS) -> list[str]:
    """Split one page's text into overlapping ~chunk_tokens pieces."""
    text = text.strip()
    if not text:
        return []
    tokens = _ENCODING.encode(text)
    if len(tokens) <= chunk_tokens:
        return [text]

    chunks = []
    step = chunk_tokens - overlap_tokens
    start = 0
    while start < len(tokens):
        piece = tokens[start : start + chunk_tokens]
        chunks.append(_ENCODING.decode(piece))
        if start + chunk_tokens >= len(tokens):
            break
        start += step
    return chunks


def build_corpus(pdf_dir: Path = PDF_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDFs found in {pdf_dir}. Run `python src/download_corpus.py` first."
        )

    for pdf_path in pdf_paths:
        document = pdf_path.stem
        pages = extract_pages(pdf_path)
        for page_num, page_text in enumerate(pages, start=1):
            for i, piece in enumerate(chunk_page_text(page_text)):
                chunk_id = f"{document}__p{page_num}__c{i}"
                chunks.append(Chunk(chunk_id=chunk_id, document=document, page=page_num, text=piece))
    return chunks


def save_corpus(chunks: list[Chunk], path: Path = CHUNKS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in chunks], f, ensure_ascii=False, indent=2)


def load_corpus(path: Path = CHUNKS_PATH) -> list[Chunk]:
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    return [Chunk(**item) for item in raw]


if __name__ == "__main__":
    chunks = build_corpus()
    save_corpus(chunks)
    print(f"Built {len(chunks)} chunks from {len(set(c.document for c in chunks))} documents")
    print(f"Saved -> {CHUNKS_PATH}")
