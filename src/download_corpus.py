"""Download the Air Products Safetygram PDFs that make up the corpus.

PDFs are not committed to the repo (see .gitignore) -- run this script to
fetch them into data/pdfs/ before running ingestion.
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "pdfs"

# Official PDF URLs, confirmed live on www.airproducts.com (2026-11-18).
SAFETYGRAMS = {
    "safetygram-7-liquid-nitrogen.pdf": (
        "https://www.airproducts.com/-/media/files/en/900/"
        "900-13-081-us-nov15-liquid-nitrogen-safetygram-7.pdf"
    ),
    "safetygram-6-liquid-oxygen.pdf": (
        "https://www.airproducts.com/-/media/files/en/900/"
        "900-13-078-us-liquid-oxygen-safetygram-6.pdf"
    ),
    "safetygram-4-gaseous-hydrogen.pdf": (
        "https://www.airproducts.com/-/media/files/en/900/"
        "900-13-095-us-gaseous-hydrogen-safetygram-4.pdf"
    ),
    "safetygram-9-liquid-hydrogen.pdf": (
        "https://www.airproducts.com/-/media/files/en/900/"
        "900-13-082-us-liquid-hydrogen-safetygram-9.pdf"
    ),
    "safetygram-16-cryogenic-liquids.pdf": (
        "https://www.airproducts.com/-/media/files/en/900/"
        "900-13-079-us-safe-handling-of-cryo-liquids-safetygram-16.pdf"
    ),
}

# Air Products' CDN rejects requests with no browser-like User-Agent.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
    )
}


def download_all() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in SAFETYGRAMS.items():
        dest = DATA_DIR / filename
        if dest.exists() and dest.stat().st_size > 0:
            print(f"skip  {filename} (already downloaded)")
            continue
        print(f"fetch {filename}")
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        print(f"  saved {len(resp.content):,} bytes -> {dest}")


if __name__ == "__main__":
    try:
        download_all()
    except requests.RequestException as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        sys.exit(1)
