# app.py – Final (RAG + caching)

import streamlit as st
import pandas as pd
import faiss
import json
import os
import sys
import importlib

st.set_page_config(page_title="AI Security Log Analyzer", layout="wide")
st.title("AI-Based Security Log Analyzer (RAG)")
st.markdown("""
Upload any **CSV / JSON / TXT** log file.
The system will:
- Normalize logs  
- Build embeddings + FAISS index  
- Analyze via RAG 
""")


# -------------------------
# File Loader
# -------------------------
def load_file(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    if file.name.endswith(".json"):
        return pd.json_normalize(json.load(file))
    if file.name.endswith(".txt"):
        lines = file.read().decode("utf-8", errors="ignore").splitlines()
        return pd.DataFrame({"log_line": lines})
    return None


def prepare_text(df):
    df = df.fillna("Unknown")
    text_cols = [c for c in df.columns if df[c].dtype == "object"]
    return df[text_cols].astype(str).agg(" | ".join, axis=1).tolist()


# -------------------------
# Embeddings + FAISS (cached)
# -------------------------
@st.cache_resource
def get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data
def create_index(df):
    emb = get_embedder()
    texts = prepare_text(df)
    v = emb.encode(texts, convert_to_numpy=True)
    dim = v.shape[1]
    idx = faiss.IndexFlatL2(dim)
    idx.add(v)
    return idx


uploaded = st.file_uploader("Upload log file", type=["csv", "json", "txt"])

if uploaded:
    df = load_file(uploaded)
    if df is None:
        st.error("Unsupported file.")
        st.stop()

    st.success(f"Loaded {df.shape[0]} rows.")
    st.dataframe(df.head())

    with st.spinner("Building FAISS index..."):
        index = create_index(df)

    # load LLM once
    with st.spinner("Loading model..."):
        sys.path.append(os.path.dirname(__file__))
        rag = importlib.import_module("rag_pipeline")
        importlib.reload(rag)

        load_llm = getattr(rag, "load_llm")
        retrieve_and_analyze = getattr(rag, "retrieve_and_analyze")
        llm = load_llm()

    q = st.text_area("Ask a question:")

    if st.button("Analyze"):
        if not q.strip():
            st.warning("Enter a query.")
        else:
            with st.spinner("Analyzing..."):
                result = retrieve_and_analyze(q, index, df, llm)

            st.markdown("## Threat Intelligence Report")
            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Suspicious IPs")
                s = result["suspicious_ips"]
                st.write(s if s else "None")

            with col2:
                st.subheader("Recurring IPs")
                s = result["recurring_ips"]
                st.write(s if s else "None")

            if result["failed_users"]:
                st.subheader("Failed Users")
                st.write(result["failed_users"])

            st.subheader("Summary")
            st.write(result["summary"])

            st.success(result["conclusion"])

            st.subheader("Relevant Logs")
            st.write(result["relevant_logs"][:25])


st.markdown("---")
st.caption("AI Security Log Analyzer")
