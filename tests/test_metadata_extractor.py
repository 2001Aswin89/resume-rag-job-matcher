from src.ingestion.metadata_extractor import (
    extract_candidate_name,
    extract_education,
    extract_experience_years,
    extract_metadata,
    extract_skills,
    split_skills,
)
from src.models import ResumeChunk, ResumeDocument


def test_extract_candidate_name_from_first_non_empty_line():
    document = ResumeDocument(
        filename="resume.txt",
        path="data/resumes/resume.txt",
        raw_text="\n\nJohn Doe\nSoftware Engineer",
    )

    assert extract_candidate_name(document) == "John Doe"


def test_extract_candidate_name_returns_unknown_for_empty_document():
    document = ResumeDocument(
        filename="resume.txt",
        path="data/resumes/resume.txt",
        raw_text="   \n\n",
    )

    assert extract_candidate_name(document) == "Unknown"


def test_split_skills_preserves_parenthesized_commas():
    skills = split_skills(
        "AWS (EC2, S3, SQS), Docker, Kubernetes"
    )

    assert skills == [
        "AWS (EC2, S3, SQS)",
        "Docker",
        "Kubernetes",
    ]


def test_extract_skills_from_skills_sections():
    chunks = [
        ResumeChunk(
            section="Technical Skills",
            text=(
                "Languages: Python, Java\n"
                "Frameworks: FastAPI, Django\n"
                "Tools: Docker, AWS (EC2, S3, SQS)"
            ),
        ),
        ResumeChunk(
            section="Experience",
            text="Python developer at Example Corp",
        ),
    ]

    skills = extract_skills(chunks)

    assert skills == [
        "Python",
        "Java",
        "FastAPI",
        "Django",
        "Docker",
        "AWS (EC2, S3, SQS)",
    ]


def test_extract_skills_removes_duplicates():
    chunks = [
        ResumeChunk(
            section="Skills",
            text=(
                "Languages: Python, Docker, Python"
            ),
        ),
        ResumeChunk(
            section="Technical Skills",
            text="Tools: Docker, FastAPI",
        ),
    ]

    skills = extract_skills(chunks)

    assert skills == [
        "Python",
        "Docker",
        "FastAPI",
    ]

def test_extract_experience_years_returns_largest_value():
    document = ResumeDocument(
        filename="resume.txt",
        path="data/resumes/resume.txt",
        raw_text=(
            "Software Engineer with 5 years of experience.\n"
            "Senior Engineer with 8 years of experience."
        ),
    )

    assert extract_experience_years(document) == 8.0


def test_extract_experience_years_returns_zero_when_missing():
    document = ResumeDocument(
        filename="resume.txt",
        path="data/resumes/resume.txt",
        raw_text="Software Engineer\nPython Developer",
    )

    assert extract_experience_years(document) == 0.0


def test_extract_education_from_education_section():
    chunks = [
        ResumeChunk(
            section="Education",
            text=(
                "Bachelor of Technology in Computer Science\n"
                "IIT Delhi | Graduated: 2016\n"
                "Master of Science in Computer Science"
            ),
        ),
        ResumeChunk(
            section="Experience",
            text="Worked as a software engineer.",
        ),
    ]

    education = extract_education(chunks)

    assert education == [
        "Bachelor of Technology in Computer Science",
        "Master of Science in Computer Science",
    ]


def test_extract_education_ignores_non_education_sections():
    chunks = [
        ResumeChunk(
            section="Experience",
            text="Bachelor of Technology graduate working at Google.",
        ),
        ResumeChunk(
            section="Education",
            text="B.Tech in Computer Science",
        ),
    ]

    education = extract_education(chunks)

    assert education == [
        "B.Tech in Computer Science",
    ]


def test_extract_metadata_combines_all_metadata():
    document = ResumeDocument(
        filename="resume.txt",
        path="data/resumes/resume.txt",
        raw_text=(
            "Jane Smith\n"
            "Senior Python Developer\n"
            "8 years of experience."
        ),
    )

    chunks = [
        ResumeChunk(
            section="Technical Skills",
            text="Languages: Python, FastAPI, PostgreSQL",
        ),
        ResumeChunk(
            section="Education",
            text="Bachelor of Technology in Computer Science",
        ),
    ]

    metadata = extract_metadata(document, chunks)

    assert metadata.candidate_name == "Jane Smith"
    assert metadata.resume_path == "data/resumes/resume.txt"
    assert metadata.skills == [
        "Python",
        "FastAPI",
        "PostgreSQL",
    ]
    assert metadata.experience_years == 8.0
    assert metadata.education == [
        "Bachelor of Technology in Computer Science"
    ]