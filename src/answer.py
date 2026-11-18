"""Answer generation via a local LLM (Ollama), with and without retrieval."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import requests

from src.retrieve import RetrievedChunk, Retriever

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
REQUEST_TIMEOUT = 300

REFUSAL_EN = "Not covered in the provided documents"
REFUSAL_AR = "هذا غير مذكور في الوثائق المتوفرة"

_ARABIC_RE = re.compile(r"[؀-ۿ]")


def detect_language(text: str) -> str:
    return "ar" if _ARABIC_RE.search(text) else "en"


@dataclass
class Citation:
    document: str
    page: int


@dataclass
class AnswerResult:
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]
    refused: bool


def _build_rag_prompt(question: str, lang: str, chunks: list[RetrievedChunk]) -> str:
    context_blocks = "\n\n".join(
        f"[{r.chunk.document}, page {r.chunk.page}]\n{r.chunk.text}" for r in chunks
    )
    if lang == "ar":
        instructions = (
            "أنت مساعد سلامة يجيب فقط بالاعتماد على المقتطفات المرجعية أدناه.\n"
            "القواعد:\n"
            "1. أجب بنفس لغة السؤال (العربية)، واكتب نص الإجابة نفسه بالحروف العربية فقط "
            "-- لا تستخدم أي حروف لاتينية أو حروف صينية/يابانية/كورية أو أي لغة أخرى داخل نص الإجابة.\n"
            "2. يجب أن تضع بعد كل جملة فيها معلومة مرجعًا بالصيغة [اسم_المستند, page رقم_الصفحة]، "
            "وهذا الاستشهاد بين الأقواس هو الاستثناء الوحيد: يُكتب اسم المستند بالأحرف اللاتينية "
            "الإنجليزية تمامًا كما ورد في عنوان المقتطف أعلاه، بدون أي تغيير أو ترجمة. "
            "مثال: [safetygram-7-liquid-nitrogen, page 1]\n"
            f'3. إذا كانت المقتطفات لا تدعم إجابة السؤال، أجب بالضبط: "{REFUSAL_AR}" ولا تضف شيئًا آخر.\n'
        )
        label_context = "المقتطفات المرجعية"
        label_question = "السؤال"
    else:
        instructions = (
            "You are a safety assistant that answers ONLY using the reference excerpts below.\n"
            "Rules:\n"
            "1. Answer in the same language as the question (English).\n"
            "2. After every sentence that states a fact, you MUST add a citation in the exact "
            "literal format [document_name, page N], copying the document name and page number "
            "exactly as they appear in the excerpt headers above. Example:\n"
            "   Liquid nitrogen is extremely cold. [safetygram-7-liquid-nitrogen, page 1]\n"
            f'3. If the excerpts do not support an answer to the question, reply with exactly: '
            f'"{REFUSAL_EN}" and nothing else.\n'
        )
        label_context = "Reference excerpts"
        label_question = "Question"

    return (
        f"{instructions}\n"
        f"{label_context}:\n{context_blocks}\n\n"
        f"{label_question}: {question}\n"
        f"Answer:"
    )


def _build_no_rag_prompt(question: str, lang: str) -> str:
    if lang == "ar":
        return (
            "أجب على السؤال التالي بنفس لغته (العربية)، بدون أي معلومات مرجعية إضافية. "
            "اكتب الإجابة بالحروف العربية فقط، بدون أي حروف لاتينية أو صينية/يابانية/كورية أو أي لغة أخرى.\n\n"
            f"السؤال: {question}\nالإجابة:"
        )
    return (
        "Answer the following question in the same language it was asked (English), "
        "with no additional reference material provided.\n\n"
        f"Question: {question}\nAnswer:"
    )


def call_ollama(
    prompt: str, model: str = OLLAMA_MODEL, host: str = OLLAMA_HOST, temperature: float = 0.1
) -> str:
    resp = requests.post(
        f"{host}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


_CITATION_RE = re.compile(r"\[\s*([^,\[\]]+?)\s*,\s*(?:page|صفحة|p\.?)?\s*(\d+)\s*\]", re.IGNORECASE)


def extract_citations(answer_text: str) -> list[Citation]:
    seen: set[tuple[str, int]] = set()
    citations: list[Citation] = []
    for match in _CITATION_RE.finditer(answer_text):
        document, page = match.group(1).strip(), int(match.group(2))
        key = (document, page)
        if key not in seen:
            seen.add(key)
            citations.append(Citation(document=document, page=page))
    return citations


def is_refusal(answer_text: str) -> bool:
    normalized = answer_text.strip()
    return REFUSAL_EN in normalized or REFUSAL_AR in normalized


_BRACKETED_RE = re.compile(r"\[[^\]]*\]")
_LATIN_LETTER_RE = re.compile(r"[A-Za-z]")
_CJK_RE = re.compile(r"[一-鿿぀-ヿ가-힣]")


def has_script_leak(answer_text: str) -> bool:
    """True if an Arabic answer contains Latin or CJK letters outside citation brackets."""
    without_citations = _BRACKETED_RE.sub("", answer_text)
    return bool(_LATIN_LETTER_RE.search(without_citations) or _CJK_RE.search(without_citations))


def answer_question(
    question: str,
    use_rag: bool,
    retriever: Retriever | None = None,
    model: str = OLLAMA_MODEL,
) -> AnswerResult:
    lang = detect_language(question)

    if use_rag:
        if retriever is None:
            retriever = Retriever()
        retrieved = retriever.retrieve(question)
        prompt = _build_rag_prompt(question, lang, retrieved)
    else:
        retrieved = []
        prompt = _build_no_rag_prompt(question, lang)

    raw_answer = call_ollama(prompt, model=model)
    refused = is_refusal(raw_answer)
    citations = [] if refused else extract_citations(raw_answer)

    return AnswerResult(
        answer=raw_answer,
        citations=citations,
        retrieved_chunks=retrieved,
        refused=refused,
    )


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    result = answer_question("What should I do if liquid nitrogen contacts my skin?", use_rag=True)
    print(result.answer)
    print("Citations:", result.citations)
