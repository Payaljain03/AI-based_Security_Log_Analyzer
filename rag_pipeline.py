# rag_pipeline.py
# RAG pipeline (Phi-3 Mini via Hugging Face) + intent detection + context filtering
# No external APIs required.

import os
import re
import ipaddress
import requests
from collections import Counter
import numpy as np
import pandas as pd
import faiss

from sentence_transformers import SentenceTransformer

# Transformers / HF model
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# LangChain wrappers
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFacePipeline

# -------------------------
# Config: change model_name if needed
# -------------------------
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"  # HF model
MAX_NEW_TOKENS = 600
TEMPERATURE = 0.2

# -------------------------
# Load embedding model (cached on import)
# -------------------------
print("Loading sentence-transformer embedding model (all-MiniLM-L6-v2)...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------
# Load the HF LLM (AutoModel) and wrap for LangChain
# This attempts device_map='auto' and falls back to CPU if needed.
# -------------------------
def load_hf_pipeline(model_name=MODEL_NAME):
    """Load HF tokenizer+model and return a text-generation pipeline."""
    # use try/except to avoid failing import if device_map not available
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=None,
            trust_remote_code=True
        )
        gen = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=False,
            return_full_text=False
        )
        return gen
    except Exception as e:
        # fallback: try CPU loading (slower)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map={"": "cpu"}, trust_remote_code=True)
        gen = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=False,
            return_full_text=False
        )
        return gen

print("Loading Phi-3 Mini instruct model (this may download weights)...")
hf_gen_pipeline = load_hf_pipeline(MODEL_NAME)
llm = HuggingFacePipeline(pipeline=hf_gen_pipeline)

# -------------------------
# Prompt template optimized for log analysis
# - short, explicit instructions
# - ask for structured output: summary, evidence bullet list
# -------------------------
prompt_template = """
You are an experienced cybersecurity analyst. You will be given:
- A short 1-3 sentence user query
- A small set of relevant log lines (context)

Task:
1) Provide a concise 1-3 sentence summary that directly answers the user's query.
2) Provide a short Evidence section: up to 5 bullet points referencing specific log lines (timestamp or text snippet).
3) Provide any immediate recommended action (1 line), or "No action needed" if none.

User Query:
{query}

Relevant Logs:
{context}

Answer format (strictly):
SUMMARY:
<Essential 1-3 sentences>

EVIDENCE:
- <bullet 1>
- <bullet 2>

ACTION:
<one line>
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["query", "context"])
chain = LLMChain(llm=llm, prompt=prompt)

# -------------------------
# Intent detection (simple rule-based, extendable)
# -------------------------
def detect_intent(query: str) -> str:
    q = query.lower()
    if "password" in q or "reset" in q or "change" in q:
        return "PASSWORD"
    if "failed" in q or "login" in q or "logon" in q or "authentication" in q:
        return "AUTH_FAILURE"
    if "ip" in q or "ip " in q or "address" in q or "access" in q or "connection" in q:
        return "NETWORK"
    if "error" in q or "warning" in q or "exception" in q:
        return "ERRORS"
    if "summary" in q or "suspicious" in q or "analyze" in q:
        return "SUMMARY"
    # default fallback
    return "GENERAL"

# -------------------------
# Context filtering per intent
# -------------------------
def filter_df_by_intent(df: pd.DataFrame, intent: str) -> pd.DataFrame:
    """
    Return a filtered DataFrame containing rows relevant to the intent.
    If filter yields empty, return original df (fallback).
    """
    if df is None or df.empty:
        return df

    lc = df.astype(str).agg(" ".join, axis=1).str.lower()

    if intent == "AUTH_FAILURE":
        keywords = ["failed", "failed login", "logon failure", "authentication failed", "login failure", "invalid password", "account lockout"]
    elif intent == "PASSWORD":
        keywords = ["password", "password changed", "password reset", "password update"]
    elif intent == "NETWORK":
        keywords = ["ip", "connection from", "connected from", "remote address", "source ip", "dst ip", "destination ip", "connection"]
    elif intent == "ERRORS":
        keywords = ["error", "exception", "traceback", "critical"]
    elif intent == "SUMMARY":
        keywords = ["error", "failed", "denied", "attack", "unauthorized", "login"]
    else:
        keywords = []  # general uses whole df

    if keywords:
        mask = lc.apply(lambda line: any(k in line for k in keywords))
        filtered = df.loc[mask]
        return filtered if not filtered.empty else df
    else:
        return df

# -------------------------
# Helper extractors
# -------------------------
SAFE_ORGS = [
    "Cloudflare", "Akamai", "Google", "Microsoft", "Amazon",
    "Level 3", "Verizon", "AT&T", "SoftLayer", "Facebook"
]

def extract_suspicious_ips(df: pd.DataFrame):
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
                except ipaddress.AddressValueError:
                    pass
    if not ips:
        return []
    unique_ips = list(set(ips))
    enriched = []
    for ip in unique_ips:
        info = enrich_ip(ip)
        if any(safe.lower() in info.lower() for safe in SAFE_ORGS):
            continue
        enriched.append(info)
    return enriched

def find_recurring_ips(df: pd.DataFrame, min_count: int = 2):
    if df is None or df.empty:
        return []
    text_data = df.astype(str).agg(" ".join, axis=1)
    ip_pattern = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
    ips = ip_pattern.findall(" ".join(text_data))
    rec = [ip for ip, count in Counter(ips).items() if count >= min_count]
    return rec

def enrich_ip(ip: str) -> str:
    try:
        if ip.startswith(("10.", "192.168.", "172.16.")):
            return f"{ip} (Private)"
        r = requests.get(f"https://ipinfo.io/{ip}/json", timeout=2)
        data = r.json()
        country = data.get("country", "Unknown")
        org = data.get("org", "Unknown")
        return f"{ip} ({country}, {org})"
    except Exception:
        return f"{ip} (Unknown)"

def extract_failed_users(df: pd.DataFrame):
    """
    Extract usernames from log lines that indicate failed authentication.
    Tries common patterns like 'user: NAME', 'username=NAME', 'Account Name: NAME'
    """
    if df is None or df.empty:
        return []
    text_data = df.astype(str).agg(" ".join, axis=1).str.lower()
    keywords = ["failed", "logon failure", "authentication failed", "invalid password", "account lockout"]
    user_pattern = re.compile(r"(?:user(?:name)?[:= ]+|account name[:= ]+)([\w\\.@-]+)", re.I)
    users = []
    for line in text_data:
        if any(k in line for k in keywords):
            matches = user_pattern.findall(line)
            for m in matches:
                users.append(m)
    return list(set(users))

# -------------------------
# Build FAISS index helper (used for any filtered df)
# -------------------------
def build_faiss_from_texts(texts: list):
    if not texts:
        return None
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

# -------------------------
# Main retrieve_and_analyze with intent routing
# -------------------------
def retrieve_and_analyze(query: str, index: faiss.IndexFlatL2, df: pd.DataFrame, top_k: int = 5):
    """
    query: user question
    index: FAISS index for the full df (unused directly if we rebuild over filtered df)
    df: original DataFrame
    """
    intent = detect_intent(query)
    filtered_df = filter_df_by_intent(df, intent)

    # Prepare text list for filtered df
    text_cols = [col for col in filtered_df.columns if filtered_df[col].dtype == "object"]
    if text_cols:
        texts = filtered_df[text_cols].astype(str).agg(" | ".join, axis=1).tolist()
    else:
        texts = filtered_df.astype(str).agg(" | ".join, axis=1).tolist()

    # Build a temporary FAISS index for the filtered context (fast enough for small slices)
    tmp = build_faiss_from_texts(texts)
    if tmp is None:
        return {
            "summary": "No text available for this query.",
            "suspicious_ips": [],
            "recurring_ips": [],
            "failed_users": [],
            "conclusion": "No data to analyze."
        }
    tmp_index, _ = tmp

    # Encode the query and search
    query_vec = embedder.encode([query], convert_to_numpy=True)
    distances, results = tmp_index.search(query_vec, top_k)

    # Defensive checks
    if results is None or len(results) == 0 or len(results[0]) == 0:
        return {
            "summary": "No relevant logs found for the query.",
            "suspicious_ips": [],
            "recurring_ips": [],
            "failed_users": [],
            "conclusion": "No data available to analyze."
        }

    # Collect matched logs (safe indexing)
    matched_idx = [i for i in results[0] if 0 <= i < len(texts)]
    matched_logs = [texts[i] for i in matched_idx] if matched_idx else []

    # Build the small context (fit into prompt)
    max_context_chars = 1200
    context = ""
    for log in matched_logs:
        if len(context) + len(log) + 1 <= max_context_chars:
            context += log + "\n"
        else:
            break

    if context.strip() == "":
        context = "No relevant log text available."

    # Invoke LLM chain
    response = chain.invoke({"query": query, "context": context})

    # Extract text from HF pipeline / chain
    if isinstance(response, dict) and "text" in response:
        summary = response["text"]
    elif isinstance(response, str):
        summary = response
    elif isinstance(response, list):
        # HF pipeline often returns list of dicts
        summary = response[0].get("generated_text", "")
    else:
        summary = "No valid response from LLM."

    # Clean the summary
    summary = re.sub(r"http\S+|www\S+|support@.+", "", summary).strip()
    # Remove stray long signatures if any
    summary = summary.split("University")[0].strip()

    # Rule-based extractions for structured outputs
    suspicious_ips = extract_suspicious_ips(filtered_df)
    recurring_ips = find_recurring_ips(filtered_df)
    failed_users = extract_failed_users(filtered_df)
    conclusion = generate_conclusion(summary, filtered_df)

    # Return structured result
    return {
        "summary": summary,
        "suspicious_ips": suspicious_ips,
        "recurring_ips": recurring_ips,
        "failed_users": failed_users,
        "conclusion": conclusion
    }

# -------------------------
# generate_conclusion uses evidence + severity heuristics (keeps same behavior)
# -------------------------
def generate_conclusion(summary: str, df: pd.DataFrame = None):
    if df is None or df.empty:
        return "No logs available for analysis."

    text_data = df.astype(str).agg(" ".join, axis=1).str.lower()
    failed_events = df[df.astype(str).apply(lambda r: "failed" in " ".join(r).lower(), axis=1)]
    suspicious_ips = extract_suspicious_ips(df)
    ip_count = len(suspicious_ips) if isinstance(suspicious_ips, list) else 0

    # Check severity based on log level if present
    if "Level" in df.columns:
        levels = df["Level"].astype(str).str.lower()
        has_critical = any("error" in lvl or "critical" in lvl for lvl in levels)
    else:
        has_critical = False

    base = summary.lower()

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

