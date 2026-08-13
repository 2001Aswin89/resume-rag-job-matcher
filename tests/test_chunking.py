from src.ingestion.chunking import chunk_resume


def test_chunk_resume_preserves_section_boundaries():
    resume = """
    John Doe

    Summary:
    Backend developer with Python experience.

    Skills:
    Python, FastAPI, PostgreSQL

    Experience:
    Built REST APIs using FastAPI.
    """

    chunks = chunk_resume(resume)

    assert len(chunks) == 3

    assert chunks[0].section == "Summary"
    assert "Backend developer with Python experience." in chunks[0].text

    assert chunks[1].section == "Skills"
    assert "Python, FastAPI, PostgreSQL" in chunks[1].text

    assert chunks[2].section == "Experience"
    assert "Built REST APIs using FastAPI." in chunks[2].text


def test_chunk_resume_without_section_headers_returns_general_chunk():
    resume = """
    John Doe
    Backend developer with Python experience.
    """

    chunks = chunk_resume(resume)

    assert len(chunks) == 1
    assert chunks[0].section == "General"
    assert "Backend developer with Python experience." in chunks[0].text