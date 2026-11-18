"""Tiny synthetic corpus for tests that must not depend on the real, gitignored
Safetygram PDFs/chunks (those are not present in a fresh checkout or CI runner).
"""
from src.ingest import Chunk

FIXTURE_CHUNKS = [
    Chunk(
        chunk_id="n2-1",
        document="safetygram-7-liquid-nitrogen",
        page=2,
        text=(
            "Liquid nitrogen is extremely cold, with a boiling point of -320.5F "
            "(-195.8C) at atmospheric pressure. Skin contact causes frostbite."
        ),
    ),
    Chunk(
        chunk_id="o2-1",
        document="safetygram-6-liquid-oxygen",
        page=8,
        text=(
            "When handling liquid oxygen, wear personal protective equipment "
            "including a face shield, insulated gloves, and safety shoes to "
            "prevent skin or eye contact with the cryogenic liquid."
        ),
    ),
    Chunk(
        chunk_id="h2-1",
        document="safetygram-4-gaseous-hydrogen",
        page=1,
        text=(
            "Gaseous hydrogen is flammable over a wide range of 4 percent to 75 "
            "percent by volume in air and can displace oxygen, causing "
            "asphyxiation in enclosed spaces."
        ),
    ),
]
