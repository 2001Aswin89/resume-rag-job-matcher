from chromadb import PersistentClient


class VectorStore:
    """
    Persistent ChromaDB wrapper for resume chunks.
    """

    def __init__(
        self,
        persist_directory: str = "chroma_db",
        collection_name: str = "resume_chunks",
    ):
        self.client = PersistentClient(path=persist_directory)

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def add_chunks(self, chunks_with_metadata: list[dict]) -> None:
        """
        Add resume chunks and their metadata to ChromaDB.

        Each item must contain:
            {
                "id": str,
                "text": str,
                "embedding": list[float],
                "metadata": dict
            }
        """
        if not chunks_with_metadata:
            return

        ids = [chunk["id"] for chunk in chunks_with_metadata]
        documents = [chunk["text"] for chunk in chunks_with_metadata]
        embeddings = [
            chunk["embedding"]
            for chunk in chunks_with_metadata
        ]
        metadatas = [
            chunk["metadata"]
            for chunk in chunks_with_metadata
        ]

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(
        self,
        embedding: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> dict:
        """
        Query ChromaDB for the most similar resume chunks.

        Args:
            embedding: Query embedding vector.
            top_k: Maximum number of results to return.
            filters: Optional ChromaDB metadata filter.
        """
        query_kwargs = {
            "query_embeddings": [embedding],
            "n_results": top_k,
        }

        if filters is not None:
            query_kwargs["where"] = filters

        return self.collection.query(**query_kwargs)