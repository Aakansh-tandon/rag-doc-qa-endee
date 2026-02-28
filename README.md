# 📄 RAG Document Q&A — Powered by Endee Vector DB & Gemini

---

## 📌 Project Overview & Problem Statement

Finding specific answers inside long PDF documents is time-consuming and tedious. Traditional keyword search often fails because the same concept can be expressed in many different ways.

**This project solves that problem** by building a **Retrieval-Augmented Generation (RAG)** web application. A user uploads any PDF document and then asks natural-language questions about it. The system:

1. **Stores** the document's content as semantic vector embeddings in **Endee**, a high-performance vector database.
2. **Retrieves** the most relevant passages using cosine similarity search.
3. **Generates** a precise, grounded answer using **Google Gemini** LLM — citing only the information present in the document.

This eliminates manual searching and leverages the power of semantic understanding to deliver accurate, context-aware answers in seconds.

---

## 🔄 System Design — Data Flow Diagrams

The system has two distinct flows:

### Flow 1 — Ingestion (When user uploads a PDF)

```
┌─────────────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   User uploads PDF                                              │
│         │                                                       │
│         ▼                                                       │
│   Extract text page-by-page                                     │
│   (PyMuPDF / fitz)                                              │
│         │                                                       │
│         ▼                                                       │
│   Split text into overlapping chunks                            │
│   (500 chars each, 50 char overlap)                             │
│         │                                                       │
│         ▼                                                       │
│   Generate 384-dimensional vector embedding                     │
│   for each chunk (sentence-transformers all-MiniLM-L6-v2)       │
│         │                                                       │
│         ▼                                                       │
│   Store each chunk's vector + metadata                          │
│   (text, page number, filename) into Endee                      │
│   Index: "documents" | Space: cosine | Precision: INT8          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Flow 2 — Question Answering (When user asks a question)

```
┌─────────────────────────────────────────────────────────────────┐
│                   QUESTION ANSWERING PIPELINE                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   User types a question                                         │
│         │                                                       │
│         ▼                                                       │
│   Generate 384-dim vector embedding for the question            │
│   (same sentence-transformers model)                            │
│         │                                                       │
│         ▼                                                       │
│   Query Endee "documents" index                                 │
│   → top 5 most similar chunks (cosine similarity)               │
│         │                                                       │
│         ▼                                                       │
│   Build RAG prompt:                                             │
│   "Answer using ONLY the context below.                         │
│    If not found, say so.                                        │
│    Context: [5 chunks]  Question: [user question]"              │
│         │                                                       │
│         ▼                                                       │
│   Send prompt to Gemini API                                     │
│         │                                                       │
│         ▼                                                       │
│   Display generated answer + source chunks                      │
│   (with page numbers and similarity scores)                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Tech Stack

| Layer            | Tool                                       | Purpose                                          |
| ---------------- | ------------------------------------------ | ------------------------------------------------ |
| Vector Database  | Endee (via Docker)                         | Store and search document embeddings              |
| Embedding Model  | sentence-transformers `all-MiniLM-L6-v2`   | Convert text to 384-dimensional vectors           |
| PDF Parser       | PyMuPDF (`fitz`)                           | Extract text from uploaded PDF files              |
| LLM              | Google Gemini (`gemini-2.0-flash`)         | Generate answers from retrieved context           |
| Frontend         | Streamlit                                  | Web UI for document upload and Q&A                |
| Language         | Python 3.10+                               | All backend and frontend logic                    |

---

## 🔌 How Endee is Specifically Used

Endee is used as the **sole vector storage and search engine** in this project. Here is exactly how each operation works:

### 1. Connection

```python
from endee import Endee

client = Endee()  # Connects to localhost:8080, no auth token for local dev
```

### 2. Index Creation

An index named `"documents"` is created with the following configuration:

| Parameter   | Value     | Why                                                     |
| ----------- | --------- | ------------------------------------------------------- |
| `dimension`   | 384       | Matches the output dimension of `all-MiniLM-L6-v2`       |
| `space_type`  | `cosine`  | Cosine similarity for semantic matching                  |
| `precision`   | `int8d`   | 8-bit integer quantization — good accuracy, low memory   |

```python
# Index is created via direct HTTP call to handle SDK/server precision mismatch
requests.post("http://localhost:8080/api/v1/index/create", json={
    "index_name": "documents",
    "dim": 384,
    "space_type": "cosine",
    "precision": "int8d",
    "M": 16,
    "ef_con": 128,
})
```

### 3. Upserting Vectors (During Ingestion)

Each PDF chunk is stored as a vector with metadata:

```python
index = client.get_index(name="documents")

index.upsert([
    {
        "id": "a1b2c3d4-...",              # UUID for each chunk
        "vector": [0.12, -0.03, ...],       # 384-dimensional embedding
        "meta": {
            "text": "The actual chunk text...",
            "page": 3,                      # Source page number
            "filename": "report.pdf"        # Original filename
        }
    },
    # ... more chunks (batched, max 500 per call)
])
```

### 4. Querying Vectors (During Q&A)

The user's question is embedded and searched:

```python
results = index.query(
    vector=[0.08, 0.15, ...],   # Question's 384-dim embedding
    top_k=5                      # Return 5 most similar chunks
)

# Each result contains:
# - id: chunk identifier
# - similarity: cosine similarity score (0 to 1)
# - meta: {"text": "...", "page": 3, "filename": "report.pdf"}
```

The top 5 results are passed as context to the Gemini LLM for answer generation.

---

## 🗂️ Project Structure

```
rag-doc-qa-endee/
│
├── app/
│   ├── __init__.py          # Python package marker
│   ├── ingestor.py          # PDF → text → chunks → embeddings → Endee upsert
│   ├── retriever.py         # Question → embedding → Endee query → ranked chunks
│   ├── generator.py         # Chunks + question → RAG prompt → Gemini → answer
│   └── pipeline.py          # Orchestrator: retriever → generator → result
│
├── ui/
│   └── streamlit_app.py     # Streamlit web interface (upload + Q&A)
│
├── docker-compose.yml       # Endee vector DB Docker container
├── .env                     # GEMINI_API_KEY (secret)
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## 🚀 Step-by-Step Setup & Execution

### Prerequisites

- **Docker Desktop** installed and running on Windows
- **Python 3.10+** installed
- A free **Gemini API key** from [Google AI Studio](https://aistudio.google.com)

### Step 1 — Start Endee Vector Database

Make sure Docker Desktop is running, then:

```bash
docker compose up -d
```

Endee is now running at `http://localhost:8080`. You can verify by opening this URL in your browser — you should see the Endee dashboard.

### Step 2 — Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

```powershell
# Windows PowerShell
.\venv\Scripts\Activate.ps1
```

### Step 3 — Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs: `endee`, `sentence-transformers`, `pymupdf`, `streamlit`, `google-genai`, `python-dotenv`

### Step 4 — Add Gemini API Key

Edit the `.env` file and replace the placeholder:

```
GEMINI_API_KEY=your-actual-gemini-api-key
```

Get a free key at [aistudio.google.com](https://aistudio.google.com).

### Step 5 — Launch the Application

```bash
streamlit run ui/streamlit_app.py
```

The app opens in your browser at `http://localhost:8501`.

### Step 6 — Use the Application

1. **Upload a PDF** from the sidebar → click **🔄 Ingest Document**
2. Wait for the success message showing how many chunks were ingested
3. **Type a question** in the main area → click **🚀 Get Answer**
4. View the AI-generated answer and expand source chunks to see the exact passages used

