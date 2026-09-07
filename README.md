# Local Standard RAG Demo

This project demonstrates a complete local Standard RAG pipeline.

## Ingestion

- Markdown document parsing
- Sliding-window chunking
- Metadata enrichment
- Sentence Transformer embeddings
- ChromaDB vector storage

## Retrieval

- Query embedding
- Dense vector retrieval
- Sparse BM25 retrieval
- Reciprocal Rank Fusion

## Generation

- Context construction
- Local LLM using Ollama
- Knowledge-grounded answer generation

## Requirements

- Python 3.11+
- Ollama
- qwen2.5:1.5b

## Installation

Create a virtual environment:

```bash
python -m venv .venv