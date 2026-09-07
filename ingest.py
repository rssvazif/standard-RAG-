from pathlib import Path
import hashlib

import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DOCUMENTS_DIR,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    OVERLAP,
)


# ---------------------------------------------------------
# 1. Document Parser
# ---------------------------------------------------------

def parse_markdown(path: Path) -> list[dict]:
    """
    Parse Markdown and preserve heading context.
    """

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    sections = []

    heading_stack = []
    current_lines = []

    def flush():
        if not current_lines:
            return

        content = "\n".join(current_lines).strip()

        if content:
            sections.append(
                {
                    "content": content,
                    "heading_path": " > ".join(heading_stack),
                }
            )

        current_lines.clear()

    for line in lines:

        stripped = line.strip()

        if stripped.startswith("#"):

            level = len(stripped) - len(stripped.lstrip("#"))

            if 1 <= level <= 6 and stripped[level:level + 1] == " ":

                flush()

                heading = stripped[level:].strip()

                heading_stack = heading_stack[: level - 1]
                heading_stack.append(heading)

                current_lines.append(stripped)

                continue

        current_lines.append(line)

    flush()

    return sections


# ---------------------------------------------------------
# 2. Sliding Window Chunking
# ---------------------------------------------------------

def sliding_window(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    # Basic normalization
    text = " ".join(text.split())

    chunks = []

    start = 0

    step = chunk_size - overlap

    while start < len(text):

        chunk = text[
            start:start + chunk_size
        ].strip()

        if chunk:
            chunks.append(chunk)

        start += step

    return chunks


# ---------------------------------------------------------
# 3. Metadata Enrichment
# ---------------------------------------------------------

def enrich_metadata(
    path: Path,
    section: dict,
    chunk_index: int,
) -> dict:

    return {
        "source": path.name,
        "source_path": str(
            path.relative_to(DOCUMENTS_DIR)
        ),
        "file_type": path.suffix.lower(),
        "heading_path": (
            section["heading_path"]
            or "root"
        ),
        "chunk_index": chunk_index,
    }


# ---------------------------------------------------------
# Stable Chunk ID
# ---------------------------------------------------------

def make_id(
    metadata: dict,
    text: str,
) -> str:

    raw = (
        f'{metadata["source_path"]}:'
        f'{metadata["chunk_index"]}:'
        f'{text}'
    )

    return hashlib.sha1(
        raw.encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------
# Ingestion Pipeline
# ---------------------------------------------------------

def main():

    # Vector DB
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_or_create_collection(
        COLLECTION_NAME
    )

    # Embedding Model
    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    documents = []
    metadatas = []
    ids = []

    markdown_files = sorted(
        DOCUMENTS_DIR.rglob("*.md")
    )

    if not markdown_files:
        raise SystemExit(
            "No Markdown files found in ./documents"
        )

    # Process every document
    for path in markdown_files:

        # 1. Parse
        sections = parse_markdown(path)

        # 2. Chunk
        for section in sections:

            chunks = sliding_window(
                section["content"],
                chunk_size=CHUNK_SIZE,
                overlap=OVERLAP,
            )

            for chunk_index, chunk in enumerate(chunks):

                # 3. Metadata
                metadata = enrich_metadata(
                    path,
                    section,
                    chunk_index,
                )

                documents.append(chunk)
                metadatas.append(metadata)

                ids.append(
                    make_id(
                        metadata,
                        chunk,
                    )
                )

    # Embedding
    embeddings = model.encode(
        documents,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).tolist()

    # 4. Store in Vector DB
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print(
        f"Indexed {len(documents)} chunks."
    )

    print(
        f"Collection: {COLLECTION_NAME}"
    )

    print(
        f"Vector DB: {CHROMA_DIR}"
    )


if __name__ == "__main__":
    main()