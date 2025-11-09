# -*- coding: utf-8 -*-
"""RAG pipeline"""

# ==============================================
# AI-Based Security Log Analyzer - RAG Pipeline
# Model: Gemini (Google Vertex AI) or fallback HF pipeline
# ==============================================

import os
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Latest LangChain imports
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
from langchain_community.llms import HuggingFacePipeline

# ================== Load FAISS Indexes and Datasets ==================
aws_index_path = "/content/drive/MyDrive/AI-based_Security_Log_Analyzer/model/aws_index.faiss"
zeek_index_path = "/content/drive/MyDrive/AI-based_Security_Log_Analyzer/model/zeek_index.faiss"

aws_df_path = "/content/drive/MyDrive/AI-based_Security_Log_Analyzer/data/final_cleaned_normalized_dataset_1aws.csv"
zeek_df_path = "/content/drive/MyDrive/AI-based_Security_Log_Analyzer/data/final_cleaned_normalized_dataset_zeek2.csv"

print("🔹 Loading FAISS indexes and datasets...")

# Check if indexes exist
if not os.path.exists(aws_index_path) or not os.path.exists(zeek_index_path):
    raise FileNotFoundError("FAISS index files not found. Please check paths.")

aws_index = faiss.read_index(aws_index_path)
zeek_index = faiss.read_index(zeek_index_path)

aws_df = pd.read_csv(aws_df_path)
zeek_df = pd.read_csv(zeek_df_path)

print("✅ Data and indexes loaded successfully.")

# ================== Load Embedding Model ==================
print("🔹 Loading embedding model (MiniLM-L6-v2)....")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ================== Initialize LLM ==================
# Option 1: Use Google Gemini (requires API key)
# from langchain.chat_models import ChatGoogleGemini
# llm = ChatGoogleGemini(model_name="gemini-1.5-turbo", temperature=0.0, api_key="YOUR_API_KEY_HERE")

# Option 2: HF GPT2 fallback (no API key required)
print("🔹 Initializing fallback GPT2 pipeline...")
hf_pipeline = pipeline("text-generation", model="gpt2", max_new_tokens=500) # Increased max_new_tokens for fuller output
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# ================== Prompt Template ==================
template = """
You are an expert cybersecurity analyst.
Analyze the following security logs and provide professional insights.

User Query:
{query}

Relevant Logs:
{context}

Summarize your findings clearly:
1. Highlight any suspicious activity
2. Identify recurring IPs or failed logins
3. Provide a concise conclusion
"""
prompt = PromptTemplate(template=template, input_variables=["query", "context"])

# Use LLMChain with HuggingFacePipeline
chain = LLMChain(llm=llm, prompt=prompt)

# ================== Retrieval + Analysis Function ==================
def retrieve_and_analyze(query, top_k=2):
    print(f"\n🔹 Processing query: {query}")

    # Convert query into vector
    query_vec = embedder.encode([query])

    # Search in FAISS indexes
    _, aws_results = aws_index.search(query_vec, top_k)
    _, zeek_results = zeek_index.search(query_vec, top_k)

    # Retrieve top matching logs
    aws_logs = aws_df.iloc[aws_results[0]].astype(str).agg(" | ".join, axis=1).tolist()
    zeek_logs = zeek_df.iloc[zeek_results[0]].astype(str).agg(" | ".join, axis=1).tolist()

    # Merge logs and limit context length to avoid exceeding model's token limit
    full_context_list = aws_logs + zeek_logs

    # GPT2 has a max context of 1024 tokens. Rough estimate: 1 token ~ 4 characters.
    # Leaving room for the prompt template and query, target context around 3000 characters.
    max_context_chars = 400 # Reduced max_context_chars further to avoid Index Error
    current_context_chars = 0
    limited_context_logs = []
    for log_entry in full_context_list:
        # +1 for newline character between logs
        if current_context_chars + len(log_entry) + 1 <= max_context_chars:
            limited_context_logs.append(log_entry)
            current_context_chars += len(log_entry) + 1
        else:
            break # Stop adding logs if we're nearing the limit

    context = "\n".join(limited_context_logs)

    # Fallback if even the first log is too long on its own
    if not limited_context_logs and full_context_list:
        context = full_context_list[0][:max_context_chars]
        print("Warning: Even a single log entry was too long, it has been truncated.")

    print("🔹 Sending logs to LLM for AI analysis...")
    response = chain.invoke({"query": query, "context": context})

    # For HF pipeline fallback
    if isinstance(response, list):
        response_text = response[0]["generated_text"]
    else:
        response_text = response["text"]

    print("✅ Analysis complete.")
    return response_text

# ================== Main Run ==================
if __name__ == "__main__":
    user_query = "Show all failed login attempts or suspicious IP addresses."
    result = retrieve_and_analyze(user_query)

    print("\n🔹 AI Security Insight:\n")
    print(result)

    # Save output
    output_path = "/content/drive/MyDrive/AI-based_Security_Log_Analyzer/model/security_analysis.txt"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write("User Query: " + user_query + "\n\n")
        f.write("AI Security Insight:\n")
        f.write(result)

    print(f"\n✅ Output saved at: {output_path}")

