"""Streamlit UI for the Bilingual Gas Safety Assistant."""
from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Bilingual Gas Safety Assistant", page_icon="🧊")

st.title("🧊 Bilingual Gas Safety Assistant")
st.caption("Demo only, not for operational safety use.")

use_rag = st.toggle("Use RAG (retrieve from Safetygram documents)", value=True)

question = st.text_area(
    "Ask a question in English or Arabic",
    placeholder="What should I do if liquid nitrogen contacts my skin? / ما هي مخاطر الهيدروجين الغازي؟",
    height=100,
)

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Thinking..."):
        try:
            resp = requests.post(
                f"{API_URL}/ask", json={"question": question, "use_rag": use_rag}, timeout=120
            )
            resp.raise_for_status()
            result = resp.json()
        except requests.RequestException as exc:
            st.error(f"Request to API failed: {exc}")
            result = None

    if result:
        st.subheader("Answer")
        st.write(result["answer"])

        if result["citations"]:
            st.subheader("Citations")
            for c in result["citations"]:
                st.markdown(f"- **{c['document']}**, page {c['page']}")

        if result["retrieved_chunks"]:
            st.subheader("Retrieved snippets")
            for chunk in result["retrieved_chunks"]:
                with st.expander(f"{chunk['document']} — page {chunk['page']} (score {chunk['score']:.3f})"):
                    st.write(chunk["text"])
