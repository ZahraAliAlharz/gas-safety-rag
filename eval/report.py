"""Compute and print/save the results table from eval/results.csv, split by
language (en/ar) and mode (RAG/no-RAG)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_PATH = EVAL_DIR / "results.csv"
REPORT_PATH = EVAL_DIR / "results.md"


def _rate(series: pd.Series, value: str = "yes") -> float | None:
    valid = series[series.isin(["yes", "no"])]
    if len(valid) == 0:
        return None
    return round((valid == value).mean(), 3)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (lang, use_rag), group in df.groupby(["lang", "use_rag"]):
        unanswerable = group[group["answerable"] == "no"]
        rows.append(
            {
                "lang": lang,
                "mode": "RAG" if use_rag else "no-RAG",
                "n": len(group),
                "accuracy": _rate(group["correct"]),
                "hallucination_rate": _rate(group["hallucinated"]),
                "citation_accuracy": _rate(group["citation_correct"]),
                "refusal_accuracy_on_unanswerable": _rate(unanswerable["correct"]),
                "retrieval_hit@3": _rate(group["hit_at_3"]) if use_rag else None,
            }
        )
    summary = pd.DataFrame(rows).sort_values(["mode", "lang"]).reset_index(drop=True)
    return summary


def to_markdown(summary: pd.DataFrame) -> str:
    lines = ["# Evaluation results", ""]
    lines.append(summary.to_markdown(index=False, floatfmt=".3f"))
    lines.append("")
    lines.append(
        "accuracy / hallucination_rate / citation_accuracy are fractions of all "
        "questions in that group (n=10 per lang x mode, 20 total per mode, 40 overall). "
        "refusal_accuracy_on_unanswerable is restricted to the 4 unanswerable questions "
        "per lang. retrieval_hit@3 is restricted to answerable questions in RAG mode."
    )
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    df = pd.read_csv(RESULTS_PATH)
    summary = build_summary(df)
    md = to_markdown(summary)
    print(md)
    REPORT_PATH.write_text(md, encoding="utf-8")
    print(f"\nSaved -> {REPORT_PATH}")
