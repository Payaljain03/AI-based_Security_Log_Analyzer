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
Upload any **CSV / JSON / XML / Text log file**, and the system will:
- Preprocess and normalize your logs  
- Create embeddings & FAISS index (cached)  
- Analyze them via your RAG pipeline  
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
    "Upload your log file (CSV, JSON, XML, or TXT)",
    type=["csv", "json", "xml", "txt"]
)

# ===============================================================
# Helper: File Processing
# ===============================================================
def load_file(file):
    """Load different log file formats into pandas DataFrame"""
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)

    elif file.name.endswith(".json"):
        df = pd.json_normalize(json.load(file))

    elif file.name.endswith(".xml"):
        tree = ET.parse(file)
        root = tree.getroot()
        records = []

        def parse_element(elem, parent_key=""):
            """Recursively flatten nested XML elements"""
            data = {}
            for child in elem:
                tag = f"{parent_key}.{child.tag}" if parent_key else child.tag
                if len(child):
                    data.update(parse_element(child, tag))
                else:
                    text = (child.text or "").strip()
                    if child.attrib:
                        for k, v in child.attrib.items():
                            data[f"{tag}_{k}"] = v
                    data[tag] = text
            return data

        for item in root.findall(".//*"):
            if len(item):
                records.append(parse_element(item))

        df = pd.DataFrame(records).replace("", None)

    elif file.name.endswith(".txt"):
        lines = file.read().decode("utf-8").splitlines()
        df = pd.DataFrame({"log_line": lines})

    else:
        st.error("Unsupported file format.")
        return None

    return df


def prepare_text(df):
    """Combine useful columns into text for embeddings."""
    df = df.fillna("Unknown")

    if "Message" in df.columns:
        combined_text = df["Message"].astype(str).tolist()
    else:
        text_cols = [col for col in df.columns if df[col].dtype == "object"]
        combined_text = df[text_cols].astype(str).agg(" | ".join, axis=1)

    return combined_text


@st.cache_resource
def get_embedding_model():
    """Load and cache embedding model."""
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data(show_spinner=False)
def create_index_from_df(df):
    """Create embeddings + FAISS index (cached for performance)."""
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
        st.success(f"File loaded successfully! {df.shape[0]} rows detected.")
        st.dataframe(df.head())

        with st.spinner("Creating embeddings & FAISS index (cached)..."):
            index = create_index_from_df(df)
        st.success("Embeddings & FAISS index ready (cached for reuse).")

        # -----------------------------------------------------------
        #  User Query Section
        # -----------------------------------------------------------
        st.markdown("Ask a Question about your Logs")
        query = st.text_area(
            "Example: 'Summarize suspicious IP activity' or 'Show failed login attempts'",
            height=100
        )

        # -----------------------------------------------------------
        # Analyze Button
        # -----------------------------------------------------------
        if st.button("Analyze Logs"):
            if not query.strip():
                st.warning("Please enter a query before analyzing.")
            else:
                with st.spinner("Analyzing with LLM + RAG pipeline..."):
                    try:
                        result = retrieve_and_analyze(query, index, df)

                        st.markdown("## Threat Intelligence Report")
                        st.markdown("---")

                        # --- Two columns for IPs and Patterns ---
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

                        # --- Main Conclusion ---
                        st.markdown("---")
                        st.subheader("Conclusion")
                        st.success(result.get("conclusion", "No conclusion generated."))

                        # --- Related Logs (only if relevant) ---
                        if "failed" in query.lower():
                            st.markdown("---")
                            st.subheader("Related Log Entries")
                            failed_logs = df[
                                df.astype(str)
                                .apply(lambda r: "failed" in " ".join(r).lower(), axis=1)
                            ]
                            st.dataframe(failed_logs.head(10))

                    except Exception as e:
                        st.error(f"Error during analysis: {e}")

# ===============================================================
# Footer
# ===============================================================
st.markdown("---")
st.caption("AI-based Security Log Analyzer (RAG + LLM)")
