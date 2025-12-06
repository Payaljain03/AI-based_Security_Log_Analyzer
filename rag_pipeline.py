# rag_pipeline.py
# RAG pipeline (Phi-3 Mini via Hugging Face) + intent detection + context filtering

import os
import re
import ipaddress
import requests
from collections import Counter
import numpy as np
import pandas as pd
import faiss

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFacePipeline


MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
MAX_NEW_TOKENS = 600
TEMPERATURE = 0.2

# -------------------------------
# FIX #1 — Remove global embedder
# -------------------------------
_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        print("Loading embedding model (MiniLM-L6-v2)...")
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


# -------------------------------
# LLM Loader
# -------------------------------
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype="auto", trust_remote_code=True
    ).to("cpu")

    text_gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        do_sample=False,
        return_full_text=False,
    )
    return HuggingFacePipeline(pipeline=text_gen)


# -------------------------------
# Prompt Template
# -------------------------------
prompt_template = """
You are an experienced cybersecurity analyst. You will be given:
- A short 1-3 sentence user query
- A small set of relevant log lines (context)

Task:
1) Provide a concise 1-3 sentence summary that directly answers the user's query.
2) Provide a short Evidence section: up to 5 bullet points referencing specific log lines.
3) Provide 1-line recommended action.

User Query:
{query}

Relevant Logs:
{context}

Answer format:
SUMMARY:
<1-3 sentences>

EVIDENCE:
- <bullet>
- <bullet>

ACTION:
<one sentence>
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["query", "context"])


# -------------------------------
# Intent detection
# -------------------------------
def detect_intent(query: str) -> str:
    q = query.lower()
    if "password" in q:
        return "PASSWORD"
    if "failed" in q or "login" in q or "authentication" in q:
        return "AUTH_FAILURE"
    if "ip" in q or "access" in q or "connection" in q:
        return "NETWORK"
    if "error" in q or "exception" in q:
        return "ERRORS"
    if "suspicious" in q or "summary" in q:
        return "SUMMARY"
    return "GENERAL"


# -------------------------------
# Context filtering
# -------------------------------
def filter_df_by_intent(df: pd.DataFrame, intent: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    lc = df.astype(str).agg(" ".join, axis=1).str.lower()

    if intent == "AUTH_FAILURE":
        keywords = ["failed", "logon failure", "authentication failed"]
    elif intent == "PASSWORD":
        keywords = ["password", "reset", "changed"]
    elif intent == "NETWORK":
        keywords = ["ip", "connection", "remote address", "source ip"]
    elif intent == "ERRORS":
        keywords = ["error", "exception", "critical"]
    elif intent == "SUMMARY":
        keywords = ["failed", "denied", "attack"]
    else:
        keywords = []

    if keywords:
        mask = lc.apply(lambda line: any(k in line for k in keywords))
        filtered = df.loc[mask]
        return filtered if not filtered.empty else df
    return df


# -------------------------------
# Extractors
# -------------------------------
def extract_suspicious_ips(df):
    if df is None or df.empty:
        return []

    text_data = df.astype(str).agg(" ".join, axis=1).str.lower()
    suspicious_keywords = ["failed login", "unauthorized", "brute", "denied", "attack"]
    ip_pattern = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")

    ips = []
    for line in text_data:
        if any(k in line for k in suspicious_keywords):
            for ip in ip_pattern.findall(line):
                try:
                    ipaddress.IPv4Address(ip)
                    ips.append(ip)
                except:
                    pass

    if not ips:
        return []

    unique_ips = list(set(ips))
    enriched = []
    for ip in unique_ips:
        enriched.append(enrich_ip(ip))

    return enriched


def find_recurring_ips(df, min_count=2):
    if df is None or df.empty:
        return []
    text_data = df.astype(str).agg(" ".join, axis=1)
    ip_pattern = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
    ips = ip_pattern.findall(" ".join(text_data))
    return [ip for ip, count in Counter(ips).items() if count >= min_count]


def enrich_ip(ip):
    try:
        if ip.startswith(("10.", "192.168.", "172.16.")):
            return f"{ip} (Private)"
        r = requests.get(f"https://ipinfo.io/{ip}/json", timeout=2)
        data = r.json()
        return f"{ip} ({data.get('country')}, {data.get('org')})"
    except:
        return f"{ip} (Unknown)"


def extract_failed_users(df):
    if df is None or df.empty:
        return []
    text_data = df.astype(str).agg(" ".join, axis=1).str.lower()

    keywords = ["failed", "authentication failed", "invalid password"]
    user_pattern = re.compile(r"(?:user[:= ]+|username[:= ]+)([\w\\.@-]+)", re.I)

    users = []
    for line in text_data:
        if any(k in line for k in keywords):
            users.extend(user_pattern.findall(line))
    return list(set(users))


# -------------------------------
# FIX #2 — use embedder from get_embedder()
# -------------------------------
def build_faiss_from_texts(texts):
    if not texts:
        return None
    embedder = get_embedder()
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings


# -------------------------------
# retrieve_and_analyze
# -------------------------------
def retrieve_and_analyze(query, index, df, llm=None, top_k=5):
    if llm is None:
        llm = load_llm()

    intent = detect_intent(query)
    filtered_df = filter_df_by_intent(df, intent)

    # combine text
    text_cols = [c for c in filtered_df.columns if filtered_df[c].dtype == "object"]
    texts = (
        filtered_df[text_cols].astype(str).agg(" | ".join, axis=1).tolist()
        if text_cols else
        filtered_df.astype(str).agg(" | ".join, axis=1).tolist()
    )

    tmp = build_faiss_from_texts(texts)
    if tmp is None:
        return {"summary": "No data.", "conclusion": "No logs."}

    tmp_index, _ = tmp
    query_vec = get_embedder().encode([query], convert_to_numpy=True)
    distances, results = tmp_index.search(query_vec, top_k)

    matched_logs = [texts[i] for i in results[0] if 0 <= i < len(texts)]
    context = "\n".join(matched_logs)[:1200] or "No relevant logs."

    chain = LLMChain(llm=llm, prompt=prompt)
    summary = chain.run({"query": query, "context": context}).strip()

    return {
        "summary": summary,
        "suspicious_ips": extract_suspicious_ips(filtered_df),
        "recurring_ips": find_recurring_ips(filtered_df),
        "failed_users": extract_failed_users(filtered_df),
        "conclusion": "Analysis completed."
    }
