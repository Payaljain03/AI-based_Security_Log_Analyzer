# rag_pipeline.py
# RAG pipeline using Google Gemma-2B-it (safe, no trust_remote_code)

import os
import re
import ipaddress
import requests
from collections import Counter
import numpy as np
import pandas as pd
import faiss

from sentence_transformers import SentenceTransformer

# HF model (Gemma)
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFacePipeline


# -------------------------
# Stable, safe model
# -------------------------
MODEL_NAME = "google/gemma-2b-it"
MAX_NEW_TOKENS = 400
TEMPERATURE = 0.1


# -------------------------
# Embedding Model
# -------------------------
print("Loading MiniLM embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# -------------------------
# LLM Loader (cached in app.py)
# -------------------------
def load_llm():
    print("Loading Gemma-2B-it...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto"
    ).to("cpu")

    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        do_sample=False
    )

    return HuggingFacePipeline(pipeline=gen)


# -------------------------
# Prompt Template
# -------------------------
prompt_template = """
You are an experienced cybersecurity analyst. You will be given:
- A user query
- A small set of relevant log lines

Your task:
1) Provide a clear 1–3 sentence summary answering the user's query
2) Give 3–5 bullet points of evidence from context
3) Give one-line recommended action

User Query:
{query}

Relevant Logs:
{context}

FORMAT:

SUMMARY:
<1–3 sentences>

EVIDENCE:
- point 1
- point 2
- point 3

ACTION:
<one line>
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["query", "context"])


# -------------------------
# Intent Detection
# -------------------------
def detect_intent(query: str) -> str:
    q = query.lower()
    if any(x in q for x in ["password", "reset", "change"]):
        return "PASSWORD"
    if any(x in q for x in ["failed", "login", "logon", "auth"]):
        return "AUTH_FAILURE"
    if any(x in q for x in ["ip", "address", "connection", "access"]):
        return "NETWORK"
    if any(x in q for x in ["error", "warning", "exception"]):
        return "ERRORS"
    if "summary" in q or "suspicious" in q:
        return "SUMMARY"
    return "GENERAL"


# -------------------------
# Filtering by Intent
# -------------------------
def filter_df_by_intent(df, intent):
    if df is None or df.empty:
        return df

    lc = df.astype(str).agg(" ".join, axis=1).str.lower()

    keywords = {
        "AUTH_FAILURE": ["failed", "invalid", "denied", "authentication"],
        "PASSWORD": ["password", "reset", "changed"],
        "NETWORK": ["ip", "connection", "connected from", "remote", "src", "dst"],
        "ERRORS": ["error", "exception", "critical"],
        "SUMMARY": ["failed", "denied", "attack", "unauthorized"]
    }.get(intent, [])

    if keywords:
        mask = lc.apply(lambda line: any(k in line for k in keywords))
        filtered = df.loc[mask]
        return filtered if not filtered.empty else df

    return df


# -------------------------
# IP + User Extractors
# -------------------------
SAFE_ORGS = ["Cloudflare", "Akamai", "Google", "Microsoft", "Amazon"]

def enrich_ip(ip):
    try:
        if ip.startswith(("10.", "172.16.", "192.168.")):
            return f"{ip} (Private)"
        r = requests.get(f"https://ipinfo.io/{ip}/json", timeout=2)
        data = r.json()
        return f"{ip} ({data.get('country', 'NA')}, {data.get('org', 'NA')})"
    except:
        return f"{ip} (Unknown)"

def extract_suspicious_ips(df):
    if df is None or df.empty:
        return []
    txt = df.astype(str).agg(" ".join, axis=1)
    pattern = r"\b\d{1,3}(?:\.\d{1,3}){3}\b"
    ips = re.findall(pattern, " ".join(txt))
    uniq = list(set(ips))
    out = []
    for ip in uniq:
        info = enrich_ip(ip)
        if not any(s.lower() in info.lower() for s in SAFE_ORGS):
            out.append(info)
    return out

def find_recurring_ips(df):
    if df is None or df.empty:
        return []
    pattern = r"\b\d{1,3}(?:\.\d{1,3}){3}\b"
    txt = df.astype(str).agg(" ".join, axis=1)
    ips = re.findall(pattern, " ".join(txt))
    c = Counter(ips)
    return [ip for ip, count in c.items() if count >= 2]

def extract_failed_users(df):
    if df is None or df.empty:
        return []
    pattern = re.compile(r"(?:user|username|account name)[:= ]+([\w\.-]+)", re.I)
    txt = df.astype(str).agg(" ".join, axis=1)
    users = []
    for line in txt:
        users += pattern.findall(line)
    return list(set(users))


# -------------------------
# FAISS Builder
# -------------------------
def build_faiss_from_texts(texts):
    if not texts:
        return None
    emb = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    dim = emb.shape[1]
    idx = faiss.IndexFlatL2(dim)
    idx.add(emb)
    return idx, emb


# -------------------------
# Main RAG
# -------------------------
def retrieve_and_analyze(query, index, df, llm):
    intent = detect_intent(query)
    filtered_df = filter_df_by_intent(df, intent)

    text_rows = filtered_df.astype(str).agg(" | ".join, axis=1).tolist()
    tmp = build_faiss_from_texts(text_rows)
    if tmp is None:
        return {"summary": "No logs.", "conclusion": "No data"}

    idx, _ = tmp
    qv = embedder.encode([query], convert_to_numpy=True)
    distances, results = idx.search(qv, 5)

    chosen = [text_rows[i] for i in results[0] if i < len(text_rows)]
    context = "\n".join(chosen)[:1200]

    chain = LLMChain(llm=llm, prompt=prompt)
    resp = chain.invoke({"query": query, "context": context})

    summary = resp.get("text", resp)

    return {
        "summary": summary,
        "suspicious_ips": extract_suspicious_ips(filtered_df),
        "recurring_ips": find_recurring_ips(filtered_df),
        "failed_users": extract_failed_users(filtered_df),
        "relevant_logs": chosen,
        "conclusion": "Analysis complete."
    }
