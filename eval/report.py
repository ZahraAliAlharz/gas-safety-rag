"""Compute and print/save the results table from eval/results.csv, split by
language (en/ar) and mode (RAG/no-RAG)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_PATH = EVAL_DIR / "results.csv"
REPORT_PATH = EVAL_DIR / "results.md"


def _rate_with_count(series: pd.Series, value: str = "yes") -> str:
    """Format a yes/no rate as 'rate (hits/total)', ignoring n/a values."""
    valid = series[series.isin(["yes", "no"])]
    if len(valid) == 0:
        return "n/a"
    hits = int((valid == value).sum())
    total = len(valid)
    rate = hits / total
    return f"{rate:.3f} ({hits}/{total})"


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (lang, use_rag), group in df.groupby(["lang", "use_rag"]):
        unanswerable = group[group["answerable"] == "no"]
        row = {
            "lang": lang,
            "mode": "RAG" if use_rag else "no-RAG",
            "n": len(group),
            "accuracy": _rate_with_count(group["correct"]),
            "hallucination_rate": _rate_with_count(group["hallucinated"]),
            "citation_accuracy": _rate_with_count(group["citation_correct"]),
            "refusal_accuracy_on_unanswerable": _rate_with_count(unanswerable["correct"]),
        }
        if use_rag:
            row["hit@3_exact_page"] = _rate_with_count(group["hit_at_3_exact_page"])
            row["hit@3_page±1"] = _rate_with_count(group["hit_at_3_page_pm1"])
            row["hit@3_doc"] = _rate_with_count(group["hit_at_3_doc"])
        else:
            row["hit@3_exact_page"] = "n/a"
            row["hit@3_page±1"] = "n/a"
            row["hit@3_doc"] = "n/a"
        if lang == "ar":
            row["script_leak_rate"] = _rate_with_count(group["script_leak"])
        else:
            row["script_leak_rate"] = "n/a"
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(["mode", "lang"]).reset_index(drop=True)
    return summary


def to_markdown(summary: pd.DataFrame) -> str:
    lines = ["# Evaluation results", ""]
    lines.append(summary.to_markdown(index=False))
    lines.append("")
    lines.append(
        "Every rate is shown as `rate (count/total)`. accuracy / hallucination_rate / "
        "citation_accuracy are out of all questions in that group (n=10 per lang x mode, "
        "20 total per mode, 40 overall). refusal_accuracy_on_unanswerable is restricted to "
        "the 4 unanswerable questions per lang. The three hit@3 columns are restricted to "
        "answerable questions in RAG mode: hit@3_exact_page requires the gold page itself "
        "among the top-3 retrieved chunks, hit@3_page±1 additionally credits a top-3 "
        "chunk from the gold document within one page of the gold page, and hit@3_doc only "
        "requires a top-3 chunk from the correct document. script_leak_rate (Arabic rows "
        "only) is the fraction of answers containing Latin or CJK letters outside citation "
        "brackets."
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
