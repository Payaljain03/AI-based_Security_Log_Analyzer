# 🔐 AI-Based Security Log Analyzer (RAG)

An **AI-powered Security Log Analysis web application** that uses **Retrieval-Augmented Generation (RAG)** to analyze security logs and generate meaningful threat intelligence using natural language queries.

This project demonstrates how **LLMs + vector search** can be applied to cybersecurity log analysis in a practical, deployable system.

---

## 🚀 Live Demo

🔗 **Deployed Application:**  
https://<your-streamlit-app-url>

---

## 🎯 Project Objective

To convert **raw security logs** into **actionable security insights** such as:
- Suspicious IP addresses
- Attack patterns
- Risk level assessment
- High-level summaries

using **AI + RAG architecture**.

---

## 🧠 End-to-End Workflow (Core of the Project)

### 1️⃣ Log Upload
- User uploads a security log file (`CSV / JSON / TXT`)
- Logs may include:
  - IP addresses
  - Timestamps
  - Events (login attempts, scans, access requests)

---

### 2️⃣ Data Preprocessing
- Logs are loaded into a Pandas DataFrame
- Each row is converted into a **structured text chunk**
- Missing or irrelevant fields are safely handled

---

### 3️⃣ Embedding Generation
- Each log entry is converted into a vector using:
  - **SentenceTransformer (`all-MiniLM-L6-v2`)**
- These embeddings represent the semantic meaning of log events

---

### 4️⃣ Vector Indexing (FAISS)
- Embeddings are stored in a **FAISS vector index**
- Enables fast semantic similarity search over logs
- Index is created **only once per uploaded file**

---

### 5️⃣ User Query Processing
- User asks a natural language question, e.g.:
  - *Summarize this log file*
  - *List suspicious IPs*
- Query is also converted into an embedding

---

### 6️⃣ Retrieval (RAG Step)
- FAISS retrieves the **most relevant log entries**
- Only the top matching logs are selected
- This prevents hallucination and keeps responses grounded in data

---

### 7️⃣ LLM Reasoning
- Retrieved log context is passed to the LLM
- Model used:
  - **TinyLlama/TinyLlama-1.1B-Chat-v1.0**
- The LLM:
  - Analyzes patterns
  - Identifies attacks
  - Generates explanations

---

### 8️⃣ Threat Intelligence Output
- Results are displayed in the UI as:
  - Attack type
  - Risk level
  - Key insights
  - Suspicious / malicious IPs
- Output is **structured and human-readable**

---

## 🖥️ Application Interface

- File upload panel
- Interactive query box
- Real-time analysis results
- Clean and minimal Streamlit UI

---

## 🛠 Tech Stack

### Core
- Python
- Streamlit
- Pandas, NumPy

### RAG Components
- FAISS (vector database)
- Sentence Transformers
  - `all-MiniLM-L6-v2`

### LLM
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Lightweight
- CPU-friendly
- Non-gated (Streamlit Cloud compatible)

---

## 💬 Example Questions You Can Ask

- Summarize this log file
- List suspicious IP addresses
- Are there any automated attacks?
- What is the risk level?
- Which IPs are repeatedly accessing the system?
- Is there any network scanning behavior?

---

## 🧪 Sample Output

- **Attack Type:** Automated attack  
- **Risk Level:** HIGH  
- **Key Insight:** Systematic network probing detected  
- **Suspicious IPs:**
  - `198.51.100.23`
  - `203.0.113.50`

---

## 🎯 Use Cases

- Cybersecurity learning projects
- SOC-style log triage
- Incident investigation
- AI + RAG portfolio demonstration
- Internship / entry-level AI roles

---

## ⚠️ Limitations

- Designed for small to medium log files
- CPU-based inference
- Not a replacement for enterprise SIEM tools

---

## 🔮 Future Improvements

- Real-time log ingestion
- Severity scoring
- Visualization dashboards
- Alert generation
- SIEM integration

---

## 👩‍💻 Author

**Payal**    

🔗 GitHub: https://github.com/Payaljain03  
🔗 LinkedIn: https://linkedin.com/in/payaljain-ml

---

⭐ If you find this project useful, give it a star on GitHub!


