"""Run all eval questions through both RAG and no-RAG modes, grade the outputs
with an LLM judge, and save everything to eval/results.csv.

Deterministic checks (no judge call needed):
  - refused:            detected from the exact refusal string
  - hit_at_3_exact_page: whether gold_page appears among the top-3 retrieved pages (RAG only)
  - hit_at_3_page_pm1:   whether a top-3 chunk from gold_doc is within 1 page of gold_page
  - hit_at_3_doc:        whether any top-3 chunk comes from gold_doc at all
  - script_leak:         for Arabic rows, whether the answer body (outside [..] citations)
                         contains Latin or CJK letters; "n/a" for English rows
  - citation_correct:   whether a citation matches (gold_doc, gold_page) -- only meaningful
                       for answerable, non-refused rows; "n/a" otherwise
  - correct/hallucinated when the question is unanswerable: refusing is correct and not
                       hallucinated; answering instead is wrong and is a hallucination

LLM-judge call (qwen2.5:3b) needed only for answerable, non-refused rows, to decide
whether the generated answer actually matches the gold answer and whether it contains
unsupported claims.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.answer import answer_question, has_script_leak
from src.retrieve import Retriever

EVAL_DIR = Path(__file__).resolve().parent
QUESTIONS_PATH = EVAL_DIR / "questions.csv"
RESULTS_PATH = EVAL_DIR / "results.csv"

JUDGE_PROMPT_TEMPLATE = """You are grading a safety-assistant's answer for correctness and hallucination.

Question: {question}
Reference (gold) answer: {gold_answer}
Model's answer: {model_answer}

Judge two things:
1. "correct": does the model's answer convey the same key facts as the reference answer
   (ignoring wording differences and extra citation brackets)? yes or no.
2. "hallucinated": does the model's answer include any specific claim (a number, fact, or
   instruction) that is NOT supported by, or contradicts, the reference answer? yes or no.

Respond with ONLY a JSON object, no other text, in exactly this form:
{{"correct": "yes", "hallucinated": "no"}}
"""

_CORRECT_FIELD_RE = re.compile(r'"?correct"?\s*:\s*"?(yes|no)"?', re.IGNORECASE)
_HALLUCINATED_FIELD_RE = re.compile(r'"?hall\w*"?\s*:\s*"?(yes|no)"?', re.IGNORECASE)


def judge_answer(question: str, gold_answer: str, model_answer: str) -> tuple[str, str]:
    from src.answer import call_ollama

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question, gold_answer=gold_answer, model_answer=model_answer
    )
    raw = call_ollama(prompt)

    correct_match = _CORRECT_FIELD_RE.search(raw)
    hallucinated_match = _HALLUCINATED_FIELD_RE.search(raw)
    correct = correct_match.group(1).lower() if correct_match else "unknown"
    hallucinated = hallucinated_match.group(1).lower() if hallucinated_match else "unknown"
    return correct, hallucinated


def run_all() -> pd.DataFrame:
    questions = pd.read_csv(QUESTIONS_PATH)
    retriever = Retriever()
    rows = []

    total = len(questions) * 2
    done = 0
    for _, q in questions.iterrows():
        for use_rag in (True, False):
            done += 1
            print(f"[{done}/{total}] {q['id']} use_rag={use_rag}", flush=True)

            result = answer_question(q["question"], use_rag=use_rag, retriever=retriever)
            answerable = q["answerable"] == "yes"
            retrieved_pages = [r.chunk.page for r in result.retrieved_chunks]
            retrieved_docs = [r.chunk.document for r in result.retrieved_chunks]

            hit_at_3_exact_page = ""
            hit_at_3_page_pm1 = ""
            hit_at_3_doc = ""
            if use_rag and answerable:
                gold_page = int(q["gold_page"])
                gold_doc = q["gold_doc"]
                same_doc_pages = [
                    page for doc, page in zip(retrieved_docs, retrieved_pages) if doc == gold_doc
                ]
                hit_at_3_exact_page = "yes" if gold_page in same_doc_pages else "no"
                hit_at_3_page_pm1 = (
                    "yes" if any(abs(p - gold_page) <= 1 for p in same_doc_pages) else "no"
                )
                hit_at_3_doc = "yes" if gold_doc in retrieved_docs else "no"

            script_leak = has_script_leak(result.answer) if q["lang"] == "ar" else None
            script_leak_label = "n/a" if script_leak is None else ("yes" if script_leak else "no")

            if not answerable:
                citation_correct = "n/a"
            elif result.refused:
                citation_correct = "no"  # should have answered with a citation but didn't
            else:
                citation_correct = "no"
                for c in result.citations:
                    if c.document == q["gold_doc"] and c.page == int(q["gold_page"]):
                        citation_correct = "yes"
                        break

            if not answerable:
                correct = "yes" if result.refused else "no"
                hallucinated = "no" if result.refused else "yes"
            elif result.refused:
                correct, hallucinated = "no", "no"
            else:
                correct, hallucinated = judge_answer(q["question"], q["gold_answer"], result.answer)

            rows.append(
                {
                    "id": q["id"],
                    "lang": q["lang"],
                    "use_rag": use_rag,
                    "question": q["question"],
                    "answerable": q["answerable"],
                    "gold_answer": q["gold_answer"],
                    "gold_doc": q["gold_doc"],
                    "gold_page": q["gold_page"],
                    "answer": result.answer,
                    "citations": "; ".join(f"{c.document} p{c.page}" for c in result.citations),
                    "retrieved_docs": "; ".join(retrieved_docs),
                    "retrieved_pages": "; ".join(str(p) for p in retrieved_pages),
                    "refused": "yes" if result.refused else "no",
                    "correct": correct,
                    "hallucinated": hallucinated,
                    "citation_correct": citation_correct,
                    "hit_at_3_exact_page": hit_at_3_exact_page,
                    "hit_at_3_page_pm1": hit_at_3_page_pm1,
                    "hit_at_3_doc": hit_at_3_doc,
                    "script_leak": script_leak_label,
                    "human_check": "",
                }
            )
            # Checkpoint after every call so a timeout doesn't lose completed work.
            pd.DataFrame(rows).to_csv(RESULTS_PATH, index=False, encoding="utf-8")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    df = run_all()
    df.to_csv(RESULTS_PATH, index=False, encoding="utf-8")
    print(f"\nSaved {len(df)} rows -> {RESULTS_PATH}")
