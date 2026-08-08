import re

from rank_bm25 import BM25Okapi

from src.retrieval.embeddings import EmbeddingService
from src.retrieval.vector_store import VectorStore


SEMANTIC_WEIGHT = 0.7
BM25_WEIGHT = 0.3
SEMANTIC_TOP_K = 20


def tokenize(text: str) -> list[str]:
    """
    Convert text into lowercase word tokens for BM25.
    """
    return re.findall(r"\b[a-zA-Z0-9+#.-]+\b", text.lower())


def extract_critical_terms(job_description: str) -> list[str]:
    """
    Extract terms from must-have or critical-skills lines.

    Critical terms are repeated in the BM25 query so that
    explicitly required skills receive additional keyword weight.
    """
    critical_lines = []

    for line in job_description.splitlines():
        lowered = line.lower()

        if (
            "must have" in lowered
            or "must-have" in lowered
            or "critical skill" in lowered
            or "critical skills" in lowered
        ):
            critical_lines.append(line)

    return tokenize(" ".join(critical_lines))


def normalize_scores(scores: list[float]) -> list[float]:
    """
    Normalize scores to the range 0-1.
    """
    if not scores:
        return []

    minimum = min(scores)
    maximum = max(scores)

    if maximum == minimum:
        return [1.0 for _ in scores]

    return [
        (score - minimum) / (maximum - minimum)
        for score in scores
    ]


class HybridSearcher:
    """
    Performs semantic + BM25 hybrid retrieval over resume chunks.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        vector_store: VectorStore | None = None,
    ):
        self.embedding_service = (
            embedding_service or EmbeddingService()
        )
        self.vector_store = (
            vector_store or VectorStore()
        )

    def search(
        self,
        job_description: str,
        top_k: int = 20,
    ) -> list[dict]:
        """
        Retrieve resume chunks using semantic and BM25 search.

        Semantic retrieval gets the initial candidate pool from Chroma.
        BM25 then scores those candidates against the job description.
        """

        query_embedding = self.embedding_service.embed_text(
            job_description
        )

        semantic_results = self.vector_store.query(
            query_embedding,
            top_k=top_k,
        )

        documents = semantic_results.get("documents", [[]])[0]
        metadatas = semantic_results.get("metadatas", [[]])[0]
        ids = semantic_results.get("ids", [[]])[0]
        distances = semantic_results.get("distances", [[]])[0]

        if not documents:
            return []

        # Chroma returns distances rather than similarity scores.
        # Convert distance into a similarity-like score.
        raw_semantic_scores = [
            1.0 / (1.0 + distance)
            for distance in distances
        ]

        semantic_scores = normalize_scores(
            raw_semantic_scores
        )

        # Build a BM25 index over the semantic candidate pool.
        tokenized_documents = [
            tokenize(document)
            for document in documents
        ]

        bm25 = BM25Okapi(tokenized_documents)

        query_tokens = tokenize(job_description)
        critical_terms = extract_critical_terms(
            job_description
        )

        # Repeat critical terms to give them additional BM25 weight.
        weighted_query = query_tokens + critical_terms

        raw_bm25_scores = bm25.get_scores(
            weighted_query
        ).tolist()

        bm25_scores = normalize_scores(
            raw_bm25_scores
        )

        results = []

        for index, document in enumerate(documents):
            hybrid_score = (
                SEMANTIC_WEIGHT * semantic_scores[index]
                + BM25_WEIGHT * bm25_scores[index]
            )

            results.append(
                {
                    "id": ids[index],
                    "text": document,
                    "metadata": metadatas[index],
                    "semantic_score": semantic_scores[index],
                    "bm25_score": bm25_scores[index],
                    "hybrid_score": hybrid_score,
                }
            )

        results.sort(
            key=lambda result: result["hybrid_score"],
            reverse=True,
        )

        return results[:top_k]