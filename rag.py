import os
import requests

from config import (
    OPENROUTER_MODEL,
    OPENROUTER_URL,
)

from retrieve import HybridRetriever


def build_context(results):
    parts = []

    for index, result in enumerate(results, start=1):
        metadata = result.metadata

        parts.append(
            f"""
[Source {index}]

File:
{metadata.get("source")}

Section:
{metadata.get("heading_path")}

Chunk:
{metadata.get("chunk_index")}

Content:
{result.text}
"""
        )

    return "\n---\n".join(parts)


def generate_answer(query: str, context: str):
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY environment variable is not set."
        )

    system_prompt = """
You are a RAG assistant.

Answer the user's question using ONLY
the provided knowledge base.

If the answer is not present in the
knowledge base, explicitly say that
the knowledge base does not contain
enough information.

Always cite the source number.
For example: [Source 1].
"""

    user_prompt = f"""
Knowledge Base:
{context}

User Question:
{query}

Answer:
"""

    response = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": OPENROUTER_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "temperature": 0.1,
        },
        timeout=120,
    )

    response.raise_for_status()

    data = response.json()

    return data["choices"][0]["message"]["content"].strip()

def main():

    retriever = HybridRetriever()

    print(
        "================================="
    )

    print(
        "Local Standard RAG"
    )

    print(
        "Type 'exit' to quit."
    )

    print(
        "=================================\n"
    )

    while True:

        query = input(
            "Question: "
        ).strip()

        if query.lower() in {
            "exit",
            "quit",
        }:
            break

        if not query:
            continue

        # ---------------------------------------------
        # Hybrid Retrieval
        # ---------------------------------------------

        (
            results,
            dense_results,
            sparse_results,
        ) = retriever.search(query)

        # ---------------------------------------------
        # Show Dense Retrieval
        # ---------------------------------------------

        print(
            "\n--- Dense Retrieval ---"
        )

        for result in dense_results:

            print(
                f"rank={result.dense_rank} | "
                f"{result.metadata['source']} | "
                f"{result.metadata['heading_path']}"
            )

        # ---------------------------------------------
        # Show Sparse Retrieval
        # ---------------------------------------------

        print(
            "\n--- Sparse Retrieval / BM25 ---"
        )

        for result in sparse_results:

            print(
                f"rank={result.sparse_rank} | "
                f"{result.metadata['source']} | "
                f"{result.metadata['heading_path']}"
            )

        # ---------------------------------------------
        # Show Fusion
        # ---------------------------------------------

        print(
            "\n--- After RRF Fusion ---"
        )

        for result in results:

            print(
                f"score={result.fusion_score:.5f} | "
                f"{result.metadata['source']} | "
                f"{result.metadata['heading_path']}"
            )

        # ---------------------------------------------
        # Build Context
        # ---------------------------------------------

        context = build_context(
            results
        )

        # ---------------------------------------------
        # Generation
        # ---------------------------------------------

        print(
            "\n--- Generated Answer ---"
        )

        try:

            answer = generate_answer(
                query,
                context,
            )

            print(answer)

        except requests.RequestException as error:

            print(
                f"Ollama request failed: {error}"
            )

            print(
                "\nMake sure Ollama is running "
                "and the model is available."
            )


if __name__ == "__main__":
    main()