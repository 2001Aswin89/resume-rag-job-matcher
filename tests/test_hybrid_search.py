from src.hybrid_search import HybridSearcher


class FakeEmbeddingService:
    def embed_text(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class FakeVectorStore:
    def query(
        self,
        embedding: list[float],
        top_k: int,
        filters=None,
    ) -> dict:
        return {
            "documents": [
                [
                    "Python FastAPI Docker",
                    "Java Spring Kubernetes",
                ]
            ],
            "metadatas": [
                [
                    {"candidate_name": "Python Candidate"},
                    {"candidate_name": "Java Candidate"},
                ]
            ],
            "ids": [
                [
                    "candidate-1",
                    "candidate-2",
                ]
            ],
            "distances": [
                [
                    0.0,
                    1.0,
                ]
            ],
        }


def test_hybrid_search_combines_semantic_and_bm25_scores():
    searcher = HybridSearcher(
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStore(),
    )

    results = searcher.search(
        "Python FastAPI Docker",
        top_k=2,
    )

    assert len(results) == 2

    for result in results:
        expected_score = (
            0.7 * result["semantic_score"]
            + 0.3 * result["bm25_score"]
        )

        assert result["hybrid_score"] == expected_score


def test_hybrid_search_ranks_strong_keyword_match_first():
    searcher = HybridSearcher(
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStore(),
    )

    results = searcher.search(
        "Python FastAPI Docker",
        top_k=2,
    )

    assert results[0]["metadata"]["candidate_name"] == (
        "Python Candidate"
    )

    assert results[0]["hybrid_score"] >= results[1]["hybrid_score"]