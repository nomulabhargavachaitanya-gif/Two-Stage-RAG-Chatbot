# Two-Stage RAG Chatbot & Evaluation Suite

A Retrieval-Augmented Generation (RAG) system built with **ChromaDB**, **Sentence Transformers**, and **Google Gemini**. The project features a two-stage retrieval pipeline (dense vector search followed by cross-encoder re-ranking) and an automated evaluation harness to quantitatively benchmark chunking strategies and generation accuracy against synthetic golden datasets.

---

## 🚀 Features

- **Document Chunking**: Flexible text slicing (word-based and character-based) with sliding-window overlap to preserve context across boundaries.
- **Two-Stage Retrieval Pipeline**:
  - **Stage 1 (The Net)**: Fast semantic vector search using ChromaDB and `all-MiniLM-L6-v2` embeddings to retrieve top-$k$ candidates.
  - **Stage 2 (The Scalpel)**: Cross-encoder re-ranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to score and pinpoint the most relevant context chunk.
- **Constrained Grounded Generation**: Prompts engineered with strict grounding constraints using Google Gemini (`gemini-3.8-flash`) to prevent hallucination.
- **Automated Benchmark Synthesis**: Generates synthetic ground-truth question-and-answer pairs from raw source text (`golden_dataset.json`).
- **Evaluation Harness**: Systematically tests different chunk sizes and measures both **Retrieval Hit Rate** and **LLM Generation Correctness**.

---

## 📁 Project Structure

```text
├── engine.py                 # Core Gemini client, text chunking & embedding utilities
├── ingest.py                 # Document ingestion script to populate ChromaDB
├── main.py                   # Interactive two-stage RAG chatbot CLI
├── generate_benchmark.py     # Synthetic Q&A test set generator using Gemini
├── evaluate.py               # Evaluation script comparing chunk sizes & accuracy
├── golden_dataset.json       # Synthesized evaluation benchmark dataset
├── sample_data.txt           # Sample source document for ingestion & testing
├── requirements.txt          # Python project dependencies
├── .env.example              # Template for required environment variables
├── .gitignore                # Git ignore rules for virtualenvs, secrets, and DBs
└── README.md                 # Project documentation
```

---

## 🛠️ Setup & Installation

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Google Gemini API key ([Get one here](https://aistudio.google.com/))

### 2. Clone the Repository
```bash
git clone <your-repo-url>
cd Project_1
```

### 3. Create and Activate a Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables
Copy `.env.example` to `.env` and enter your Gemini API key:
```bash
cp .env.example .env
```
Edit `.env`:
```ini
GEMINI_API_KEY="your_actual_gemini_api_key"
```

---

## 📖 Usage Guide

### 1. Ingest Documents into Vector Database
Process and store your document (`sample_data.txt`) into ChromaDB:
```bash
python ingest.py
```

### 2. Run the Interactive RAG Chatbot
Ask questions grounded strictly in the ingested document:
```bash
python main.py
```

### 3. (Optional) Generate a Synthetic Evaluation Benchmark
Synthesize new question-and-answer pairs based on your source data:
```bash
python generate_benchmark.py
```

### 4. Run Quantitative Evaluation
Benchmark different chunk sizes (e.g., 15, 30, 50 words) to measure hit rate and factual correctness:
```bash
python evaluate.py
```

---

## 🔒 Security Note
Never commit `.env` or any sensitive API keys to source control. Ensure `.env` is listed in your `.gitignore` before pushing changes.

