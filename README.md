# RAG Document Q&A

A Retrieval-Augmented Generation app that lets users upload PDFs and ask questions with page-level citations.

## Features

- PDF text extraction
- Sentence-aware chunking
- FAISS vector search
- OpenRouter embeddings
- LLM answer generation
- Source citations
- Duplicate document detection
- Evaluation script

## Tech Stack

- Streamlit
- FAISS
- SQLite
- OpenRouter
- pdfplumber
- pypdf

## Setup

- bash
  conda create -n rag python=3.11 -y
  conda activate rag
  pip install -r requirements.txt
  cp .env.example .env
  streamlit run app.py

## Evaluation

- bash
  python scripts/eval.py dl.pdf questions.json

## Known Limitations

- Scanned PDFs require OCR
- Reranking endpoint must be verified
- Evaluation currently uses keyword matching
