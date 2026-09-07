from dataclasses import dataclass

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    DENSE_TOP_K,
    SPARSE_TOP_K,
    FINAL_TOP_K,
)


@dataclass
class Result:

    id: str
    text: str
    metadata: dict

    dense_rank: int | None = None
    sparse_rank: int | None = None

    fusion_score: float = 0.0


class HybridRetriever:

    def __init__(self):

        # Vector DB
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )

        self.collection = (
            self.client.get_collection(
                COLLECTION_NAME
            )
        )

        # Embedding model
        self.model = SentenceTransformer(
            EMBEDDING_MODEL
        )

        # Load chunks for BM25
        data = self.collection.get(
            include=[
                "documents",
                "metadatas",
            ]
        )

        self.ids = data["ids"]
        self.documents = data["documents"]
        self.metadatas = data["metadatas"]

        tokenized_documents = [
            document.lower().split()
            for document in self.documents
        ]

        self.bm25 = BM25Okapi(
            tokenized_documents
        )

    # -----------------------------------------------------
    # Dense Retrieval
    # -----------------------------------------------------

    def dense_search(
        self,
        query: str,
        k: int,
    ):

        # Query → Vector
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        ).tolist()[0]

        # Vector Search
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        results = []

        for i in range(
            len(result["ids"][0])
        ):

            results.append(
                Result(
                    id=result["ids"][0][i],
                    text=result["documents"][0][i],
                    metadata=result["metadatas"][0][i],
                    dense_rank=i + 1,
                )
            )

        return results

    # -----------------------------------------------------
    # Sparse Retrieval / BM25
    # -----------------------------------------------------

    def sparse_search(
        self,
        query: str,
        k: int,
    ):

        scores = self.bm25.get_scores(
            query.lower().split()
        )

        ranked_indexes = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:k]

        results = []

        for rank, index in enumerate(
            ranked_indexes
        ):

            results.append(
                Result(
                    id=self.ids[index],
                    text=self.documents[index],
                    metadata=self.metadatas[index],
                    sparse_rank=rank + 1,
                )
            )

        return results

    # -----------------------------------------------------
    # Reciprocal Rank Fusion
    # -----------------------------------------------------

    @staticmethod
    def rrf_fusion(
        dense_results,
        sparse_results,
        k=60,
    ):

        merged = {}

        # Dense ranking
        for result in dense_results:

            if result.id not in merged:
                merged[result.id] = result

            merged[result.id].fusion_score += (
                1 /
                (k + result.dense_rank)
            )

        # Sparse ranking
        for result in sparse_results:

            if result.id not in merged:
                merged[result.id] = result

            merged[result.id].fusion_score += (
                1 /
                (k + result.sparse_rank)
            )

        # Sort by fused score
        return sorted(
            merged.values(),
            key=lambda r: r.fusion_score,
            reverse=True,
        )

    # -----------------------------------------------------
    # Hybrid Search
    # -----------------------------------------------------

    def search(self, query: str):

        dense_results = self.dense_search(
            query,
            DENSE_TOP_K,
        )

        sparse_results = self.sparse_search(
            query,
            SPARSE_TOP_K,
        )

        fused_results = self.rrf_fusion(
            dense_results,
            sparse_results,
        )

        return (
            fused_results[:FINAL_TOP_K],
            dense_results,
            sparse_results,
        )