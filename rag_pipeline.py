# rag_pipeline.py
# RAG pipeline using TinyLlama-1.1B-Chat (ungated, works on Streamlit Cloud)

import os
import re
import ipaddress
import requests
from collections import Counter
import pandas as pd
import faiss

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFacePipeline


MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_NEW_TOKENS = 350
TEMPERATURE = 0.2


print("Loading MiniLM embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def load_llm():
    print("Loading TinyLlama 1.1B Chat...")
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
        do_sample=False,
        return_full_text=False
    )

    return HuggingFacePipeline(pipeline=gen)

prompt_template = """
You are a cybersecurity analyst. Using the logs, produce:

SUMMARY:
1–3 sentences answering the query.

EVIDENCE:
- 3–5 bullets with specific log references.

ACTION:
One line.

User Query:
{query}

Relevant Logs:
{context}
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["query", "context"]
)


def detect_intent(q):
    q = q.lower()
    if "password" in q: return "PASSWORD"
    if "failed" in q or "login" in q: return "AUTH"
    if "ip" in q or "access" in q: return "NETWORK"
    if "error" in q: return "ERROR"
    return "GENERAL"


def filter_df(df, intent):
    txt = df.astype(str).agg(" ".join, axis=1).str.lower()
    keywords = {
        "PASSWORD": ["password", "reset", "changed"],
        "AUTH": ["failed", "invalid", "denied", "login"],
        "NETWORK": ["ip", "src", "dst", "remote", "connected"],
        "ERROR": ["error", "exception", "critical"],
    }.get(intent, [])

    if not keywords:
        return df

    mask = txt.apply(lambda line: any(k in line for k in keywords))
    filtered = df.loc[mask]
    return filtered if not filtered.empty else df


def extract_ips(df):
    txt = df.astype(str).agg(" ".join, axis=1)
    ips = re.findall(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", " ".join(txt))
    return list(set(ips))


def build_faiss(texts):
    if not texts:
        return None
    emb = embedder.encode(texts, convert_to_numpy=True)
    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(emb)
    return index, emb


def retrieve_and_analyze(query, index, df, llm):
    intent = detect_intent(query)
    fdf = filter_df(df, intent)

    texts = fdf.astype(str).agg(" | ".join, axis=1).tolist()
    tmp = build_faiss(texts)

    if tmp is None:
        return {"summary": "No logs", "conclusion": "No data"}

    fa, emb = tmp
    qv = embedder.encode([query], convert_to_numpy=True)
    _, res = fa.search(qv, 5)

    chosen = [texts[i] for i in res[0] if i < len(texts)]
    ctx = "\n".join(chosen)[:1200]

    chain = LLMChain(llm=llm, prompt=prompt)
    out = chain.invoke({"query": query, "context": ctx})
    summary = out.get("text", out)

    return {
        "summary": summary,
        "suspicious_ips": extract_ips(fdf),
        "recurring_ips": [],
        "failed_users": [],
        "relevant_logs": chosen,
        "conclusion": "Analysis complete."
    }
