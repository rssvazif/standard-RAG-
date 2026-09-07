from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

DOCUMENTS_DIR = PROJECT_ROOT / "documents"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"


COLLECTION_NAME = "knowledge_base"


# Embedding model
EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


# Local LLM
OLLAMA_MODEL = "qwen2.5:1.5b"
OLLAMA_URL = "http://localhost:11434/api/generate"


# Chunking
CHUNK_SIZE = 500
OVERLAP = 100


# Retrieval
DENSE_TOP_K = 8
SPARSE_TOP_K = 8
FINAL_TOP_K = 5