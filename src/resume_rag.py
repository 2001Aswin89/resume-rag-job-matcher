from pathlib import Path

from src.ingestion.chunking import chunk_resume
from src.ingestion.loaders import load_documents
from src.ingestion.metadata_extractor import extract_metadata
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.vector_store import VectorStore


RESUME_DIRECTORY = Path("data/resumes")


def build_chunk_records(
    document,
    chunks,
    metadata,
    embeddings,
) -> list[dict]:
    """
    Build ChromaDB-ready records for all chunks belonging to one resume.
    """
    records = []

    for index, (chunk, embedding) in enumerate(
        zip(chunks, embeddings)
    ):
        chunk_id = (
            f"{metadata.resume_path}::chunk-{index}"
        )

        chunk_metadata = {
            "candidate_name": metadata.candidate_name,
            "resume_path": metadata.resume_path,
            "skills": ", ".join(metadata.skills),
            "experience_years": metadata.experience_years,
            "education": " | ".join(metadata.education),
            "section": chunk.section,
        }

        records.append(
            {
                "id": chunk_id,
                "text": chunk.text,
                "embedding": embedding,
                "metadata": chunk_metadata,
            }
        )

    return records


def ingest_resumes(
    resume_directory: str | Path = RESUME_DIRECTORY,
) -> tuple[int, int]:
    """
    Ingest all supported resumes into the local ChromaDB store.

    Returns:
        (resume_count, chunk_count)
    """
    documents = load_documents(resume_directory)

    embedding_service = EmbeddingService()
    vector_store = VectorStore()

    total_chunks = 0

    for document in documents:
        chunks = chunk_resume(document.raw_text)

        if not chunks:
            continue

        metadata = extract_metadata(
            document,
            chunks,
        )

        chunk_texts = [
            chunk.text
            for chunk in chunks
        ]

        embeddings = embedding_service.embed_texts(
            chunk_texts
        )

        records = build_chunk_records(
            document,
            chunks,
            metadata,
            embeddings,
        )

        vector_store.add_chunks(records)

        total_chunks += len(records)

        print(
            f"Indexed {document.filename}: "
            f"{len(records)} chunks"
        )

    return len(documents), total_chunks


def main() -> None:
    """
    Command-line entry point for resume ingestion.
    """
    print("Starting resume ingestion...")
    print(f"Resume directory: {RESUME_DIRECTORY}")
    print()

    resume_count, chunk_count = ingest_resumes()

    print()
    print("Ingestion complete.")
    print(f"Resumes ingested: {resume_count}")
    print(f"Chunks indexed: {chunk_count}")


if __name__ == "__main__":
    main()