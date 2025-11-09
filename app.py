# app.py
import streamlit as st
import pandas as pd
import numpy as np
import faiss
import json
import xml.etree.ElementTree as ET
import os
import sys
  


# ----------------------------
# App Title and Description
# ----------------------------
st.set_page_config(page_title="AI Security Log Analyzer", layout="wide")

st.title("AI-based Security Log Analyzer (RAG + LLM)")
st.markdown("""
Upload any **CSV / JSON / XML / Text log file**, and the system will:
- Preprocess and normalize your logs  
- Create embeddings & FAISS index  
- Analyze them via your RAG pipeline  
- Return summarized insights, suspicious activity & conclusions  
""")

# ----------------------------
# File Upload
# ----------------------------
uploaded_file = st.file_uploader(
    "📤 Upload your log file (CSV, JSON, XML, or TXT)",
    type=["csv", "json", "xml", "txt"]
)

# ----------------------------
# Helper: File Processing
# ----------------------------

def load_file(file):
    """Load different log file formats into pandas DataFrame"""
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    elif file.name.endswith(".json"):
        df = pd.json_normalize(json.load(file))
    elif file.name.endswith(".xml"):
        tree = ET.parse(file)
        root = tree.getroot()
        data = [{child.tag: child.text for child in elem} for elem in root]
        df = pd.DataFrame(data)
    elif file.name.endswith(".txt"):
        lines = file.read().decode("utf-8").splitlines()
        df = pd.DataFrame({"log_line": lines})
    else:
        st.error("Unsupported file format.")
        return None
    return df

def prepare_text(df):
    """Combine useful columns into text for embeddings."""
    df = df.fillna('Unknown')

    # prioritize columns that actually hold log text
    if "Message" in df.columns:
        combined_text = df["Message"].astype(str).tolist()
    else:
        text_cols = [col for col in df.columns if df[col].dtype == 'object']
        combined_text = df[text_cols].astype(str).agg(' | '.join, axis=1)
    
    return combined_text

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# ----------------------------
# Main Workflow
# ----------------------------
if uploaded_file is not None:
    sys.path.append(os.path.dirname(__file__))
  
    from sentence_transformers import SentenceTransformer
    from rag_pipeline import retrieve_and_analyze

    @st.cache_resource
    def load_embedding_model():
        return SentenceTransformer('all-MiniLM-L6-v2')
      
    with st.spinner("🔍 Reading and processing your log file..."):
        df = load_file(uploaded_file)

    if df is not None:
        st.success(f"✅ File loaded successfully! {df.shape[0]} rows detected.")
        st.write(df.head())

        # Generate embeddings dynamically
        st.info("Creating embeddings and FAISS index...")
        model = load_embedding_model()
        texts = prepare_text(df)
        embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        index = build_faiss_index(embeddings)
        st.success("✅ Embeddings & FAISS index created successfully!")

        # ----------------------------
        # User Query Section
        # ----------------------------
        st.markdown("### 💬 Ask a Question about your Logs")
        query = st.text_area(
            "Example: 'Summarize suspicious IP activity' or 'Show failed login attempts'",
            height=100
        )

        if st.button("Analyze Logs 🔎"):
            if not query.strip():
                st.warning("Please enter a query before analyzing.")
            else:
                with st.spinner("🤖 Analyzing with LLM + RAG pipeline..."):
                    try:
                        result = retrieve_and_analyze(query, index, df)  # your RAG function
                        st.caption("Automated analysis powered by AI.")
                        st.subheader("🚨 Suspicious IP Addresses")
                        st.write(result.get("suspicious_ips", "None detected."))

                        st.subheader("🔁 Recurring IPs or Patterns")
                        st.write(result.get("recurring_ips", "None found."))

                        st.subheader("🧩 Concise Conclusion")
                        st.success(result.get("conclusion", "No conclusion generated."))

                    except Exception as e:
                        st.error(f"Error during analysis: {e}")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("AI-based Security Log Analyzer (RAG + LLM)")
