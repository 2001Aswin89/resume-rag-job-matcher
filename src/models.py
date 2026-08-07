from dataclasses import dataclass, field


@dataclass(slots=True)
class ResumeDocument:
    """
    Represents a loaded resume or job description.
    """
    filename: str
    path: str
    raw_text: str


@dataclass(slots=True)
class ResumeChunk:
    """
    Represents one logical section of a resume.
    """
    section: str
    text: str


@dataclass(slots=True)
class ResumeMetadata:
    """
    Metadata extracted from an entire resume.
    """
    candidate_name: str
    resume_path: str
    skills: list[str] = field(default_factory=list)
    experience_years: float = 0.0
    education: list[str] = field(default_factory=list)