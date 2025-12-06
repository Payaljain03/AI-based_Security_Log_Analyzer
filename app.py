# =============================================================== 
# AI-Based Security Log Analyzer (RAG + LLM) - FIXED FOR DEPLOYMENT
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
import faiss
import json
import xml.etree.ElementTree as ET
import os
import sys
import importlib

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="AI Security Log Analyzer", layout="wide")
st.title("AI-based Security Log Analyzer (RAG + LLM)")

st.markdown("""
Upload any **CSV / JSON / TXT log file**, and the system will:
- Preprocess and normalize your logs
- Create embeddings & FAISS index (**cached**)
- Analyze them via the RAG pipeline (**Phi-3 Mini**)
- Return summarized insights, suspicious activity & conclusions
""")

# ===============================================================
# File Upload
# ===============================================================

uploaded_file = st.file_uploader(
    "Upload your log file (CSV, JSON or TXT)",
    type=["csv", "json", "txt"]
)

# ===============================================================
# Helper: File Processing
# ===============================================================

def load_file(file):
    """Load CSV / JSON / TXT into pandas DataFrame."""
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    elif file.name.endswith(".json"):
        return pd.json_normalize(json.load(file))
    elif file.name.endswith(".txt"):
        lines = file.read().decode("utf-8", errors="ignore").splitlines()
        return pd.DataFrame({"log_line": lines})
    else:
        st.error("Unsupported file format.")
        return None

def prepare_text(df: pd.DataFrame):
    """Combine text columns into list of strings for embedding."""
    df = df.fillna("Unknown")
    text_cols = [col for col in df.columns if df[col].dtype == "object"]
    if not text_cols:
        return df.astype(str).agg(" | ".join, axis=1).tolist()
    return df[text_cols].astype(str).agg(" | ".join, axis=1).tolist()

# -------------------------
# Cached embedding model + FAISS creation
# -------------------------

@st.cache_resource
def get_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data(show_spinner=False)
def create_index_from_df(df: pd.DataFrame):
    """Create embeddings + FAISS index (cached per DataFrame content)."""
    model = get_embedding_model()
    texts = prepare_text(df)
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# =============================================================== 
# Main Workflow
# ===============================================================

if uploaded_file is not None:
    # 1) Read file (fast)
    with st.spinner("Reading and parsing your log file..."):
        df = load_file(uploaded_file)
        if df is None:
            st.error("Could not read uploaded file.")
            st.stop()
        st.success(f"File loaded successfully — {df.shape[0]} rows detected.")
        st.dataframe(df.head())

    # 2) Create / reuse embeddings & FAISS index (cached)
    with st.spinner("Creating embeddings & FAISS index (cached)..."):
        index = create_index_from_df(df)
        st.success("Embeddings & FAISS index ready (cached).")

    # 3) Set HF token if present
    if "hf_token" in st.secrets:
        os.environ["HUGGINGFACE_HUB_API_TOKEN"] = st.secrets["hf_token"]

    # -----------------------------------------------------------
    # 4) Load RAG pipeline + Phi-3 Mini (CRASH-PROOF: cached + error handling)
    # -----------------------------------------------------------
    
    @st.cache_resource
    def get_rag_and_llm():
        """Load RAG pipeline and LLM once, cache forever."""
        try:
            sys.path.append(os.path.dirname(__file__))
            rag = importlib.import_module("rag_pipeline")
            importlib.reload(rag)
            retrieve_and_analyze = getattr(rag, "retrieve_and_analyze")
            load_llm = getattr(rag, "load_llm")
            llm = load_llm()
            return retrieve_and_analyze, llm
        except Exception as e:
            st.error(f"RAG Pipeline Load Error: {str(e)}")
            st.error("Check: 1) rag_pipeline.py in same folder 2) HF token in secrets")
            st.stop()
            return None, None

    with st.spinner("Loading RAG pipeline + Phi-3 Mini (first time only)..."):
        retrieve_and_analyze, llm = get_rag_and_llm()
        st.success("RAG pipeline & Phi-3 Mini loaded successfully! 🎉")

    # -----------------------------------------------------------
    # 5) Query + Analysis
    # -----------------------------------------------------------
    st.markdown("## Ask a Question about your Logs")
    query = st.text_area(
        "Example: 'Which IPs tried to access the system repeatedly?' or 'Show failed login attempts'",
        height=120
    )

    if st.button("Analyze Logs", type="primary"):
        if not query.strip():
            st.warning("Please enter a query before analyzing.")
        else:
            with st.spinner("Analyzing with LLM + RAG pipeline..."):
                try:
                    result = retrieve_and_analyze(query, index, df, llm=llm)
                    
                    st.markdown("## **Threat Intelligence Report**")
                    st.markdown("---")

                    # Suspicious / recurring IPs
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Suspicious IPs")
                        ips = result.get("suspicious_ips", [])
                        if ips:
                            st.table(pd.DataFrame({"IP / Geo": ips}))
                        else:
                            st.info("No suspicious IPs detected.")
                    
                    with col2:
                        st.subheader("Recurring IPs")
                        rec = result.get("recurring_ips", [])
                        if rec:
                            st.table(pd.DataFrame({"IP": rec}))
                        else:
                            st.info("No recurring IPs detected.")

                    # Failed users
                    failed_users = result.get("failed_users", [])
                    if failed_users:
                        st.markdown("---")
                        st.subheader("Users with Failed Logins")
                        st.table(pd.DataFrame({"Username": failed_users}))

                    # Summary + conclusion
                    st.markdown("---")
                    st.subheader("Summary / Conclusion")
                    st.markdown(result.get("summary", ""))
                    st.success(result.get("conclusion", "No conclusion generated."))

                    # Relevant logs
                    st.markdown("---")
                    st.subheader("Relevant Log Entries")
                    relevant = result.get("relevant_logs", None)
                    if relevant:
                        st.dataframe(pd.DataFrame({"log": relevant}).head(25))
                    else:
                        mask = df.astype(str).apply(
                            lambda r: any(word in " ".join(r).lower() for word in query.lower().split()), 
                            axis=1
                        )
                        st.dataframe(df[mask].head(25))

                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")

# =============================================================== 
# Footer
# ===============================================================

st.markdown("---")
st.caption("AI-based Security Log Analyzer (RAG + Phi-3 Mini)")
