import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

# ================= PAGE CONFIG =================
st.set_page_config(page_title="AI Security Log Analyzer", layout="wide")
st.title("AI-Based Security Log Analyzer (RAG)")

# ================= SESSION STATE =================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "texts" not in st.session_state:
    st.session_state.texts = None
    st.session_state.embeddings = None

# ================= FILE LOADER =================
def load_file(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    if file.name.endswith(".json"):
        return pd.json_normalize(json.load(file))
    if file.name.endswith(".txt"):
        lines = file.read().decode("utf-8", errors="ignore").splitlines()
        return pd.DataFrame({"log": lines})
    return None

def prepare_text(df):
    df = df.fillna("Unknown")
    text_cols = [c for c in df.columns if df[c].dtype == "object"]
    return df[text_cols].astype(str).agg(" | ".join, axis=1).tolist()

# ================= EMBEDDINGS =================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def build_embeddings(texts):
    model = load_model()
    return model.encode(texts, convert_to_numpy=True)

def retrieve_logs(query, texts, embeddings, top_k=8):
    model = load_model()
    q_emb = model.encode([query])
    scores = cosine_similarity(q_emb, embeddings)[0]
    top_idx = scores.argsort()[-top_k:][::-1]
    return [texts[i] for i in top_idx]

# ================= INTENT DETECTION (UPDATED) =================
def detect_intent(question: str) -> str:
    q = question.lower()

    if any(k in q for k in ["ip", "address"]):
        return "ip"

    if any(k in q for k in ["human", "automated", "bot"]):
        return "attack_type"

    if any(k in q for k in ["risk", "severity", "24 hours", "continue"]):
        return "risk"

    if any(k in q for k in ["why", "reason"]):
        return "reason"

    if any(k in q for k in ["fix", "mitigation", "solution", "what should i do"]):
        return "fix"

    # 🔥 CORRELATION FIX HERE
    if any(k in q for k in [
        "same campaign",
        "same attack",
        "related",
        "connected",
        "linked",
        "part of",
        "coordinated"
    ]):
        return "correlation"

    if any(k in q for k in ["what happened", "summary", "analyze"]):
        return "summary"

    return "general"

# ================= ANALYSIS HELPERS =================
def analyze_events(logs):
    text = " ".join(logs).lower()
    return {
        "network_scan": "network_scan" in text,
        "failed_login": "login | failed" in text,
        "account_locked": "account locked" in text,
        "file_access": "file_access" in text
    }

def classify_attack(flags):
    if flags["network_scan"]:
        return "Automated attack"
    if flags["failed_login"] or flags["account_locked"]:
        return "Likely automated login abuse"
    return "Possibly human-driven activity"

def assign_risk(flags):
    if flags["network_scan"] and flags["failed_login"]:
        return "HIGH"
    if flags["failed_login"] or flags["account_locked"]:
        return "MEDIUM"
    return "LOW"

def build_reason(flags):
    if flags["network_scan"]:
        return "Systematic network probing suggests automated tools scanning for open ports or weaknesses."
    if flags["failed_login"]:
        return "Multiple authentication failures in a short period indicate credential guessing or brute-force attempts."
    if flags["file_access"]:
        return "Unauthorized file access attempts suggest insufficient permissions or misuse."
    return "The behavior deviates from normal patterns observed in the logs."

def build_fixes(flags):
    fixes = []
    if flags["network_scan"]:
        fixes.append("Block the source IP and enable IDS/IPS alerts for scanning patterns.")
    if flags["failed_login"]:
        fixes.append("Enable rate limiting and account lockout; add MFA for authentication.")
    if flags["file_access"]:
        fixes.append("Review permissions and enforce role-based access control.")
    if not fixes:
        fixes.append("Continue monitoring and set alerts for anomalies.")
    return fixes

# ================= IP ANALYSIS =================
def extract_malicious_ips(logs):
    ip_reasons = {}
    for log in logs:
        parts = log.split("|")
        if len(parts) < 6:
            continue
        ip = parts[2].strip()
        event = parts[3].strip().lower()
        details = parts[5].lower()

        if "network_scan" in event:
            ip_reasons[ip] = "Network scanning behavior detected"
        elif "suspicious location" in details or "unknown location" in details:
            ip_reasons[ip] = "Login attempt from a suspicious location"
        elif "account locked" in details:
            ip_reasons[ip] = "Account locked after repeated failed logins"
    return ip_reasons

# ================= SIDEBAR =================
with st.sidebar:
    st.subheader("Upload Log File")
    uploaded = st.file_uploader("Upload CSV / JSON / TXT", type=["csv", "json", "txt"])
    if uploaded:
        df = load_file(uploaded)
        st.success(f"Loaded {len(df)} rows")
        texts = prepare_text(df)
        st.session_state.texts = texts
        st.session_state.embeddings = build_embeddings(texts)

# ================= CHAT HISTORY =================
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(chat["question"])
    with st.chat_message("assistant"):
        st.markdown(chat["answer"])

# ================= CHAT INPUT =================
user_question = st.chat_input("Ask anything related to the uploaded logs...")

if user_question:
    if not st.session_state.texts:
        st.warning("Please upload a log file first.")
    else:
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing logs..."):
                relevant_logs = retrieve_logs(
                    user_question,
                    st.session_state.texts,
                    st.session_state.embeddings
                )

            intent = detect_intent(user_question)
            flags = analyze_events(relevant_logs)

            if intent == "ip":
                ip_map = extract_malicious_ips(relevant_logs)
                if not ip_map:
                    answer = "No clearly malicious IPs were identified from the uploaded logs."
                else:
                    lines = [f"- **{ip}**: {reason}" for ip, reason in ip_map.items()]
                    answer = f"""### 🔍 Malicious IP Analysis
{chr(10).join(lines)}"""

            elif intent == "attack_type":
                answer = f"""### 🤖 Attack Classification
**Assessment:** {classify_attack(flags)}
**Explanation:** The classification is based on behavioral patterns such as systematic scanning and repeated automated actions observed in the logs."""

            elif intent == "risk":
                risk_level = assign_risk(flags)
                reasons = []

                if flags["network_scan"]:
                    reasons.append("continuous network scanning activity")
                if flags["failed_login"]:
                    reasons.append("repeated failed authentication attempts")
                if flags["account_locked"]:
                    reasons.append("account lock events due to unauthorized access attempts")

                explanation = (
                    "; ".join(reasons)
                    if reasons
                    else "no strong malicious indicators beyond minor anomalies"
                )

                answer = f"""### ⚠️ Risk Assessment

**Risk Level:** {risk_level}

**Explanation:**  
The risk is assessed as **{risk_level}** because the logs show {explanation}.  
If this behavior continues over an extended period, it increases the likelihood of system compromise or service disruption.
"""

            elif intent == "reason":
                answer = f"""### ❓ Root Cause
{build_reason(flags)}"""

            elif intent == "fix":
                fixes = build_fixes(flags)
                answer = f"""### 🛠 Recommended Actions
{chr(10).join([f"- {f}" for f in fixes])}"""

            elif intent == "correlation":
                answer = """### 🔗 Campaign Correlation
The activities may be related based on timing and behavior patterns; however, the available evidence is not sufficient to conclusively confirm a single coordinated attack campaign."""

            elif intent == "summary":
                details = []
                if flags["network_scan"]:
                    details.append("network scanning behavior")
                if flags["failed_login"]:
                    details.append("repeated failed login attempts")
                if flags["file_access"]:
                    details.append("unauthorized file access attempts")

                summary_text = ", ".join(details) if details else "unusual system activity"

                answer = f"""### 📄 Summary
Suspicious activity was detected in the uploaded logs, including {summary_text}, which deviates from normal system behavior."""

            else:
                answer = f"""### 🔍 Analysis Result
**Attack Type:** {classify_attack(flags)}
**Risk Level:** {assign_risk(flags)}
**Key Insight:** {build_reason(flags)}"""

            st.markdown(answer)

        st.session_state.chat_history.append({
            "question": user_question,
            "answer": answer
        })
