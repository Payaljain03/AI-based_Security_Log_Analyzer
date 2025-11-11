# -*- coding: utf-8 -*-
""" RAG Pipeline for AI-Based Security Log Analyzer """

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import re
import ipaddress
import requests
from collections import Counter

# -----------------------------
#  LangChain Imports (v0.3.7+ compatible)
# -----------------------------
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFacePipeline  # fixed import

# -----------------------------
# Load Embedding Model
# -----------------------------
print("Loading embedding model (MiniLM-L6-v2)...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
#  Initialize LLM
# -----------------------------
print("Initializing fallback GPT2 pipeline...")
hf_pipeline = pipeline(
    "text-generation",
    model="gpt2",
    max_new_tokens=400,
    temperature=0.7
)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# -----------------------------
#  Prompt Template
# -----------------------------
template = """
You are a professional cybersecurity analyst.
Analyze the following security logs and provide expert insights.

User Query:
{query}

Relevant Logs:
{context}

Summarize your findings clearly:
1. Identify suspicious activity (failed logins, unusual IPs, etc.)
2. List any recurring IPs or suspicious patterns.
3. Provide a short, clear conclusion.
"""
prompt = PromptTemplate(template=template, input_variables=["query", "context"])
chain = LLMChain(llm=llm, prompt=prompt)

# ===============================================================
# Core Function: retrieve_and_analyze
# ===============================================================
def retrieve_and_analyze(query, index, df, top_k=3):
    print(f"\nProcessing query: {query}")

    # Convert query into embedding vector
    query_vec = embedder.encode([query])

    # Search in FAISS index
    distances, results = index.search(query_vec, top_k)

    # --- Defensive Checks ---
    if results is None or len(results[0]) == 0:
        print("No matching logs found for query.")
        return {
            "summary": "No relevant logs found for the query.",
            "suspicious_ips": [],
            "recurring_ips": [],
            "conclusion": "No data available to analyze."
        }

    valid_results = [i for i in results[0] if 0 <= i < len(df)]
    if not valid_results:
        print("FAISS returned out-of-range indexes.")
        return {
            "summary": "Error: FAISS index mismatch with dataset.",
            "suspicious_ips": [],
            "recurring_ips": [],
            "conclusion": "Rebuild FAISS index and retry."
        }

    # Retrieve relevant logs safely
    matched_logs = (
        df.iloc[valid_results]
        .astype(str)
        .agg(" | ".join, axis=1)
        .tolist()
    )

    if not matched_logs:
        print("No relevant log entries found.")
        return {
            "summary": "No log entries found for the given query.",
            "suspicious_ips": [],
            "recurring_ips": [],
            "conclusion": "No suspicious activity detected."
        }

    # Limit context size to prevent GPT2 overflow
    max_context_chars = 800
    context = ""
    for log in matched_logs:
        if len(context) + len(log) + 1 <= max_context_chars:
            context += log + "\n"
        else:
            break

    print("Sending logs to LLM for analysis...")
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

    # Clean output
    summary = re.sub(r"http\S+|www\S+|support@.+", "", summary)
    summary = summary.strip().split("University")[0]

    print("Analysis complete.")
    
    return {
        "summary": summary,
        "suspicious_ips": extract_suspicious_ips(df),
        "recurring_ips": find_recurring_ips(df),
        "conclusion": generate_conclusion(summary, df)
    }

# ===============================================================
# Helper Functions 
# ===============================================================

SAFE_ORGS = [
    "Cloudflare", "Akamai", "Google", "Microsoft", "Amazon",
    "Level 3", "Verizon", "AT&T", "SoftLayer", "Facebook"
]

def extract_suspicious_ips(df):
    """Extract valid IPv4s related to suspicious or failed events and enrich them."""
    text_data = df.astype(str).agg(" ".join, axis=1).str.lower()
    suspicious_keywords = ["failed login", "unauthorized", "brute", "denied", "attack"]
    ip_pattern = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")

    ips = []
    for line in text_data:
        # Only capture if truly security-related
        if any(k in line for k in suspicious_keywords):
            for ip in ip_pattern.findall(line):
                try:
                    ipaddress.IPv4Address(ip)
                    ips.append(ip)
                except ipaddress.AddressValueError:
                    pass

    if not ips:
        return "No suspicious IPs detected."

    unique_ips = list(set(ips))
    enriched = []
    for ip in unique_ips:
        info = enrich_ip(ip)
        # Skip safe infrastructure (Cloudflare, AWS, etc.)
        if any(safe.lower() in info.lower() for safe in SAFE_ORGS):
            continue
        enriched.append(info)

    return enriched if enriched else "No suspicious IPs detected."

def find_recurring_ips(df):
    """Find IPs that occur multiple times."""
    text_data = df.astype(str).agg(" ".join, axis=1)
    ip_pattern = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
    ips = ip_pattern.findall(" ".join(text_data))
    rec = [ip for ip, count in Counter(ips).items() if count > 1]
    return rec if rec else "No recurring IPs."

def enrich_ip(ip):
    """Fetch basic geo + ASN info for IP, skip private ones."""
    try:
        if ip.startswith(("10.", "192.168.", "172.16.")):
            return f"{ip} (Private Network)"
        r = requests.get(f"https://ipinfo.io/{ip}/json", timeout=2)
        data = r.json()
        country = data.get("country", "Unknown")
        org = data.get("org", "Unknown")
        return f"{ip} ({country}, {org})"
    except Exception:
        return f"{ip} (Unknown)"

def generate_conclusion(summary, df=None):
    """Generate accurate, context-aware conclusions with evidence."""
    if df is None or df.empty:
        return "No logs available for analysis."

    text_data = df.astype(str).agg(" ".join, axis=1).str.lower()
    failed_events = df[df.astype(str).apply(lambda r: "failed" in " ".join(r).lower(), axis=1)]
    suspicious_ips = extract_suspicious_ips(df)
    ip_count = len(suspicious_ips) if isinstance(suspicious_ips, list) else 0

    # Check severity based on log level
    if "Level" in df.columns:
        levels = df["Level"].astype(str).str.lower()
        has_critical = any("error" in lvl or "critical" in lvl for lvl in levels)
    else:
        has_critical = False

    base = summary.lower()

    # Decision logic (context + evidence)
    if ip_count > 0 and has_critical:
        return f"Critical: {ip_count} suspicious IPs detected with high-severity events."
    elif ip_count > 0:
        return f"Possible malicious activity detected — {ip_count} suspicious IPs identified."
    elif not failed_events.empty and has_critical:
        return f"Multiple failed operations found ({len(failed_events)}). Investigate service integrity."
    elif "attack" in base:
        return "Possible attack patterns mentioned, verify manually."
    else:
        return "Normal system activity detected. No major threats found."
