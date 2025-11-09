# -*- coding: utf-8 -*-
""" RAG Pipeline for AI-Based Security Log Analyzer """

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ✅ Updated LangChain imports (0.3.7 compatible)
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import HuggingFacePipeline

# ================== Load Embedding Model ==================
print("🔹 Loading embedding model (MiniLM-L6-v2)....")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ================== Initialize LLM ==================
print("🔹 Initializing fallback GPT2 pipeline...")
hf_pipeline = pipeline(
    "text-generation",
    model="gpt2",
    max_new_tokens=400,   # balanced for Streamlit response speed
    temperature=0.7
)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# ================== Prompt Template ==================
template = """
You are a professional cybersecurity analyst.
Analyze the given security logs and provide expert insights.

User Query:
{query}

Relevant Logs:
{context}

Summarize your findings clearly:
1. Identify suspicious activity (e.g., failed logins, unusual IPs)
2. List any recurring IPs or patterns
3. Provide a concise conclusion
"""
prompt = PromptTemplate(template=template, input_variables=["query", "context"])

# Use LLMChain for generating analysis
chain = LLMChain(llm=llm, prompt=prompt)

# ===============================================================
# 🔹 Core Function: retrieve_and_analyze
# Takes in user query, FAISS index, and DataFrame dynamically.
# ===============================================================
def retrieve_and_analyze(query, index, df, top_k=3):
    print(f"\n🔹 Processing query: {query}")

    # Convert query into embedding vector
    query_vec = embedder.encode([query])

    # Retrieve top matches from FAISS index
    distances, results = index.search(query_vec, top_k)
    matched_logs = df.iloc[results[0]].astype(str).agg(" | ".join, axis=1).tolist()

    # Limit context size to prevent GPT2 overflow
    max_context_chars = 800
    context = ""
    for log in matched_logs:
        if len(context) + len(log) + 1 <= max_context_chars:
            context += log + "\n"
        else:
            break

    if not context:
        context = "No relevant logs found for this query."

    print("🔹 Sending logs to LLM for analysis...")
    response = chain.invoke({"query": query, "context": context})

    # Extract response text safely
    if isinstance(response, dict) and "text" in response:
        summary = response["text"]
    elif isinstance(response, str):
        summary = response
    elif isinstance(response, list):
        summary = response[0].get("generated_text", "")
    else:
        summary = "No valid response from LLM."

    print("✅ Analysis complete.")

    # Basic structured output
    return {
        "summary": summary.strip(),
        "suspicious_ips": extract_suspicious_ips(df),
        "recurring_ips": find_recurring_ips(df),
        "conclusion": generate_conclusion(summary)
    }

# ===============================================================
# 🔹 Helper Functions (basic rule-based insights)
# ===============================================================
def extract_suspicious_ips(df):
    """Find IPs linked with suspicious keywords (e.g., failed, denied, attack)."""
    if "source_ip" not in df.columns:
        return "No source_ip column found."

    suspicious_keywords = ["failed", "denied", "unauthorized", "attack"]
    mask = df.apply(lambda row: any(k in str(row).lower() for k in suspicious_keywords), axis=1)
    ips = df.loc[mask, "source_ip"].unique().tolist()
    return ips if ips else "No suspicious IPs detected."

def find_recurring_ips(df):
    """Detect IPs that appear multiple times."""
    if "source_ip" not in df.columns:
        return "No source_ip column found."
    recurring = df["source_ip"].value_counts()
    rec_ips = recurring[recurring > 1].index.tolist()
    return rec_ips if rec_ips else "No recurring IPs."

def generate_conclusion(text):
    """Produce a concise conclusion from the AI summary."""
    if "attack" in text.lower():
        return "Possible malicious activity detected. Further investigation recommended."
    elif "failed login" in text.lower():
        return "Multiple failed login attempts observed. Potential brute-force pattern."
    else:
        return "No major anomalies detected based on the provided logs."

# ===============================================================
# 🔹 Example (for local debugging)
# ===============================================================
if __name__ == "__main__":
    # Example dummy run
    import faiss
    dummy_data = pd.DataFrame({
        "timestamp": ["t1", "t2"],
        "source_ip": ["192.168.1.1", "192.168.1.2"],
        "event": ["login failed", "port scan detected"]
    })
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeds = model.encode(dummy_data.astype(str).agg(" | ".join, axis=1), convert_to_numpy=True)
    dim = embeds.shape[1]
    idx = faiss.IndexFlatL2(dim)
    idx.add(embeds)
    res = retrieve_and_analyze("suspicious login", idx, dummy_data)
    print(res)
