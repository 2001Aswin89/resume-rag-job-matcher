from pathlib import Path

import pdfplumber


SUPPORTED_EXTENSIONS = {".txt", ".pdf"}


def load_text_file(file_path: Path) -> str:
    """Read a UTF-8 text file."""
    return file_path.read_text(encoding="utf-8").strip()


def load_pdf_file(file_path: Path) -> str:
    """Extract text from a PDF using pdfplumber."""
    pages = []

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)

    return "\n".join(pages).strip()


def load_document(file_path: Path) -> dict:
    """
    Load a single resume or job description.

    Returns:
        {
            "filename": "...",
            "path": "...",
            "raw_text": "..."
        }
    """
    suffix = file_path.suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix}")

    if suffix == ".txt":
        text = load_text_file(file_path)
    else:
        text = load_pdf_file(file_path)

    return {
        "filename": file_path.name,
        "path": file_path.as_posix(),
        "raw_text": text,
    }


def load_documents(folder_path: str | Path) -> list[dict]:
    """
    Load every supported document from a folder.
    """
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(folder)

    documents = []

    for file_path in sorted(folder.iterdir()):
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        documents.append(load_document(file_path))

    return documents