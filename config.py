from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

DOCUMENTS_DIR = PROJECT_ROOT / "documents"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"


COLLECTION_NAME = "knowledge_base"


# Embedding model
EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


# LLM
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "openrouter/free"

# Chunking
CHUNK_SIZE = 500
OVERLAP = 100


# Retrieval
DENSE_TOP_K = 8
SPARSE_TOP_K = 8
FINAL_TOP_K = 5