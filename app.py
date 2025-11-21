# ===============================================================
# AI-Based Security Log Analyzer (RAG + LLM)
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
import faiss
import json
import xml.etree.ElementTree as ET
import os
import sys

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="AI Security Log Analyzer", layout="wide")

st.title("AI-based Security Log Analyzer (RAG + LLM)")
st.markdown("""
Upload any **CSV / JSON / Text log file**, and the system will:
- Preprocess and normalize your logs  
- Create embeddings & FAISS index (cached)  
- Analyze them via the RAG pipeline  
- Return summarized insights, suspicious activity & conclusions  
""")

# ===============================================================
#  Import pipeline globally 
# ===============================================================
sys.path.append(os.path.dirname(__file__))
from rag_pipeline import retrieve_and_analyze
from sentence_transformers import SentenceTransformer

# ===============================================================
#  File Upload
# ===============================================================
uploaded_file = st.file_uploader(
    "Upload your log file (CSV, JSON or TXT)",
    type=["csv", "json", "txt"]
)

# ===============================================================
# Helper: File Processing
# ===============================================================
def load_file(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)

    elif file.name.endswith(".json"):
        return pd.json_normalize(json.load(file))

    elif file.name.endswith(".txt"):
        lines = file.read().decode("utf-8").splitlines()
        return pd.DataFrame({"log_line": lines})

    st.error("Unsupported file format.")
    return None


def prepare_text(df):
    df = df.fillna("Unknown")
    text_cols = [col for col in df.columns if df[col].dtype == "object"]
    return df[text_cols].astype(str).agg(" | ".join, axis=1)


@st.cache_resource
def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data(show_spinner=False)
def create_index_from_df(df):
    model = get_embedding_model()
    texts = prepare_text(df)
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# ===============================================================
#  Main Workflow
# ===============================================================
if uploaded_file is not None:
    with st.spinner("Reading and processing your log file..."):
        df = load_file(uploaded_file)

    if df is not None:
        st.success(f"File loaded successfully — {df.shape[0]} rows detected.")
        st.dataframe(df.head())

        with st.spinner("Creating embeddings & FAISS index (cached)..."):
            index = create_index_from_df(df)

        st.success("Embeddings & FAISS index ready for analysis.")

        st.markdown("## Ask a Question about your Logs")
        query = st.text_area(
            "Example: 'Which IPs tried to access the system repeatedly?' or 'Show failed users'",
            height=100
        )

        if st.button("Analyze Logs"):
            if not query.strip():
                st.warning("Please enter a query before analyzing.")
            else:
                with st.spinner("Analyzing with LLM + RAG pipeline..."):
                    try:
                        result = retrieve_and_analyze(query, index, df)

                        # -------------------------------
                        # Threat Intelligence Report
                        # -------------------------------
                        st.markdown("## **Threat Intelligence Report**")
                        st.markdown("---")

                        col1, col2 = st.columns(2)

                        with col1:
                            st.subheader("Suspicious IPs")
                            ips = result.get("suspicious_ips", "None")
                            if isinstance(ips, list):
                                st.table(pd.DataFrame({"IP / Geo": ips}))
                            else:
                                st.info(ips)

                        with col2:
                            st.subheader("Recurring IPs")
                            rec = result.get("recurring_ips", "None")
                            if isinstance(rec, list):
                                st.table(pd.DataFrame({"IP": rec}))
                            else:
                                st.info(rec)

                        # Failed Users
                        failed_users = result.get("failed_users", [])
                        if isinstance(failed_users, list) and len(failed_users) > 0:
                            st.markdown("---")
                            st.subheader("Users with Failed Logins")
                            st.table(pd.DataFrame({"Username": failed_users}))

                        # Final conclusion
                        st.markdown("---")
                        st.subheader("Conclusion")
                        st.success(result.get("conclusion", "No conclusion found."))

                        # Auto display relevant logs
                        st.markdown("---")
                        st.subheader("Relevant Log Entries")
                        mask = df.astype(str).apply(
                            lambda r: any(word in " ".join(r).lower()
                                          for word in query.lower().split()),
                            axis=1
                        )
                        st.dataframe(df[mask].head(25))

                    except Exception as e:
                        st.error(f"Error during analysis: {e}")

# ===============================================================
# Footer
# ===============================================================
st.markdown("---")
st.caption("AI-based Security Log Analyzer (RAG + LLM)")
