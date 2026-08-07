import re

from src.models import ResumeChunk, ResumeDocument, ResumeMetadata


EDUCATION_KEYWORDS = (
    "b.tech",
    "b.e",
    "bachelor",
    "m.tech",
    "m.e",
    "master",
    "phd",
    "b.sc",
    "m.sc",
    "bca",
    "mca",
)

EXPERIENCE_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\+?\s+years?",
    re.IGNORECASE,
)


def extract_candidate_name(document: ResumeDocument) -> str:
    """
    Assumes the candidate name is the first non-empty line.
    """
    for line in document.raw_text.splitlines():
        line = line.strip()
        if line:
            return line

    return "Unknown"


def extract_skills(chunks: list[ResumeChunk]) -> list[str]:
    """
    Extract skills from sections containing 'skills'.
    """
    skills = []

    for chunk in chunks:
        if "skill" not in chunk.section.lower():
            continue

        for line in chunk.text.splitlines():
            line = line.strip("-• ").strip()

            if ":" in line:
                _, value = line.split(":", 1)

                for skill in value.split(","):
                    skill = skill.strip()

                    if skill:
                        skills.append(skill)
            else:
                if line:
                    skills.append(line)

    # Remove duplicates while preserving order
    return list(dict.fromkeys(skills))


def extract_experience_years(document: ResumeDocument) -> float:
    """
    Extract the largest years-of-experience value mentioned.
    """
    matches = EXPERIENCE_PATTERN.findall(document.raw_text)

    if not matches:
        return 0.0

    return max(float(value) for value in matches)


def extract_education(chunks: list[ResumeChunk]) -> list[str]:
    education = []

    for chunk in chunks:
        if "education" not in chunk.section.lower():
            continue

        for line in chunk.text.splitlines():
            line = line.strip()

            if not line:
                continue

            lower = line.lower()

            if any(keyword in lower for keyword in EDUCATION_KEYWORDS):
                education.append(line)

    return education


def extract_metadata(
    document: ResumeDocument,
    chunks: list[ResumeChunk],
) -> ResumeMetadata:
    return ResumeMetadata(
        candidate_name=extract_candidate_name(document),
        resume_path=document.path,
        skills=extract_skills(chunks),
        experience_years=extract_experience_years(document),
        education=extract_education(chunks),
    )