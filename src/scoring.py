import re
from collections import defaultdict


SEMANTIC_WEIGHT = 0.60
SKILL_WEIGHT = 0.30
EXPERIENCE_WEIGHT = 0.10


def normalize_score(score: float) -> float:
    """
    Convert a 0-1 score into a 0-100 score.
    """
    return max(0.0, min(100.0, score * 100))


def extract_required_skills(job_description: str) -> list[str]:
    """
    Extract recognizable technical skills from the
    Required Skills section using a curated vocabulary.
    """
    skill_vocabulary = [
        "Python",
        "FastAPI",
        "Django",
        "Flask",
        "PostgreSQL",
        "MySQL",
        "MongoDB",
        "Redis",
        "Docker",
        "Kubernetes",
        "AWS",
        "Azure",
        "GCP",
        "REST API",
        "GraphQL",
        "React",
        "Vue",
        "Angular",
        "Node.js",
        "TypeScript",
        "JavaScript",
        "Java",
        "Go",
        "C++",
        "C#",
        "Spring Boot",
        "Kafka",
        "RabbitMQ",
        "CI/CD",
        "Git",
        "GitHub",
        "GitLab",
        "Jenkins",
        "Terraform",
        "Ansible",
        "Linux",
        "SQL",
    ]

    required_section = []
    capture = False

    for line in job_description.splitlines():
        stripped = line.strip()

        if not stripped:
            continue

        lowered = stripped.lower()

        if "required skills" in lowered:
            capture = True
            continue

        if capture:
            # Stop at the next named section.
            if (
                lowered.startswith("preferred skills")
                or lowered.startswith("must have")
                or lowered.startswith("experience required")
                or lowered.startswith("education")
                or lowered.startswith("responsibilities")
            ):
                break

            required_section.append(stripped)

    text = " ".join(required_section)

    found = []

    for skill in sorted(
        skill_vocabulary,
        key=len,
        reverse=True,
    ):
        pattern = rf"(?<![\w+#]){re.escape(skill)}(?![\w+#])"

        if re.search(
            pattern,
            text,
            re.IGNORECASE,
        ):
            found.append(skill)

    # Preserve vocabulary order rather than
    # regex discovery order.
    vocabulary_lower = {
        skill.lower(): skill
        for skill in skill_vocabulary
    }

    return [
        vocabulary_lower[skill.lower()]
        for skill in skill_vocabulary
        if skill.lower() in {
            found_skill.lower()
            for found_skill in found
        }
    ]

def extract_must_have_skills(
    job_description: str,
    required_skills: list[str],
) -> list[str]:
    """
    Extract technical skills explicitly mentioned in the
    Must Have Requirements section.

    Skills are detected independently from required_skills so
    that a skill mentioned only in the must-have section, such
    as AWS or REST API, is still recognized.
    """
    skill_vocabulary = [
        "Python",
        "FastAPI",
        "Django",
        "Flask",
        "PostgreSQL",
        "MySQL",
        "MongoDB",
        "Redis",
        "Docker",
        "Kubernetes",
        "AWS",
        "Azure",
        "GCP",
        "REST API",
        "GraphQL",
        "React",
        "Vue",
        "Angular",
        "Node.js",
        "TypeScript",
        "JavaScript",
        "Java",
        "Go",
        "C++",
        "C#",
        "Spring Boot",
        "Kafka",
        "RabbitMQ",
        "CI/CD",
        "Git",
        "GitHub",
        "GitLab",
        "Jenkins",
        "Terraform",
        "Ansible",
        "Linux",
        "SQL",
    ]

    must_have_lines = []
    capture = False

    for line in job_description.splitlines():
        stripped = line.strip()

        if not stripped:
            continue

        lowered = stripped.lower()

        if "must have requirements" in lowered:
            capture = True
            continue

        if capture:
            if (
                lowered.startswith("experience required")
                or lowered.startswith("education")
                or lowered.startswith("preferred skills")
                or lowered.startswith("responsibilities")
            ):
                break

            must_have_lines.append(stripped)

    text = " ".join(must_have_lines)

    found = []

    # Check longer/more specific skills first so that
    # "CI/CD" is detected rather than incorrectly matching "CI".
    for skill in sorted(
        skill_vocabulary,
        key=len,
        reverse=True,
    ):
        pattern = rf"(?<![\w+#]){re.escape(skill)}(?![\w+#])"

        if re.search(
            pattern,
            text,
            re.IGNORECASE,
        ):
            found.append(skill)

    # Return skills in vocabulary order.
    found_lower = {
        skill.lower()
        for skill in found
    }

    return [
        skill
        for skill in skill_vocabulary
        if skill.lower() in found_lower
    ]

def calculate_skill_overlap(
    candidate_skills: list[str],
    required_skills: list[str],
) -> tuple[float, list[str]]:
    """
    Calculate the proportion of required skills found
    in the candidate's skills.
    """
    if not required_skills:
        return 0.0, []

    candidate_text = " ".join(candidate_skills).lower()

    matched = []

    for skill in required_skills:
        if skill.lower() in candidate_text:
            matched.append(skill)

    overlap = len(matched) / len(required_skills)

    return overlap, matched


def extract_required_experience(
    job_description: str,
) -> float | None:
    """
    Extract the minimum years-of-experience requirement.
    """
    patterns = [
        r"(\d+(?:\.\d+)?)\+?\s*years",
        r"minimum\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*years",
    ]

    values = []

    for pattern in patterns:
        matches = re.findall(
            pattern,
            job_description.lower(),
        )

        values.extend(
            float(match)
            for match in matches
        )

    if not values:
        return None

    return max(values)


def calculate_experience_fit(
    candidate_years: float,
    required_years: float | None,
) -> float:
    """
    Calculate experience fit as a 0-1 score.
    """
    if required_years is None:
        return 1.0

    if required_years == 0:
        return 1.0

    if candidate_years >= required_years:
        return 1.0

    return candidate_years / required_years


def aggregate_candidates(
    search_results: list[dict],
) -> list[dict]:
    """
    Combine multiple retrieved chunks belonging to
    the same candidate.
    """
    candidates = defaultdict(
        lambda: {
            "candidate_name": "",
            "resume_path": "",
            "skills": [],
            "experience_years": 0.0,
            "education": [],
            "chunks": [],
            "semantic_scores": [],
            "hybrid_scores": [],
        }
    )

    for result in search_results:
        metadata = result["metadata"]
        candidate_name = metadata["candidate_name"]

        candidate = candidates[candidate_name]

        candidate["candidate_name"] = candidate_name
        candidate["resume_path"] = metadata["resume_path"]

        candidate["experience_years"] = float(
            metadata.get("experience_years", 0)
        )

        candidate["skills"] = [
            skill.strip()
            for skill in metadata.get(
                "skills",
                "",
            ).split(",")
            if skill.strip()
        ]

        candidate["education"] = [
            education.strip()
            for education in metadata.get(
                "education",
                "",
            ).split("|")
            if education.strip()
        ]

        candidate["chunks"].append(
            {
                "text": result["text"],
                "section": metadata["section"],
                "hybrid_score": result["hybrid_score"],
            }
        )

        candidate["semantic_scores"].append(
            result["semantic_score"]
        )

        candidate["hybrid_scores"].append(
            result["hybrid_score"]
        )

    return list(candidates.values())


def apply_must_have_filter(
    candidates: list[dict],
    must_have_skills: list[str],
    required_years: float | None,
) -> list[dict]:
    """
    Apply hard must-have requirements.

    A candidate is excluded if they fail any explicit
    must-have skill or experience requirement.
    """
    filtered = []

    for candidate in candidates:
        _, matched_skills = calculate_skill_overlap(
            candidate["skills"],
            must_have_skills,
        )

        # Every explicit must-have skill must be present.
        if len(matched_skills) != len(must_have_skills):
            continue

        # Minimum experience is also a hard requirement.
        if required_years is not None:
            if candidate["experience_years"] < required_years:
                continue

        candidate["must_have_skills"] = matched_skills

        filtered.append(candidate)

    return filtered


def score_candidates(
    candidates: list[dict],
    required_skills: list[str],
    required_years: float | None,
) -> list[dict]:
    """
    Calculate final 0-100 candidate scores.
    """
    scored = []

    for candidate in candidates:
        semantic_score = max(
            candidate["semantic_scores"],
            default=0.0,
        )

        skill_overlap, matched_skills = (
            calculate_skill_overlap(
                candidate["skills"],
                required_skills,
            )
        )

        experience_fit = calculate_experience_fit(
            candidate["experience_years"],
            required_years,
        )

        final_score = (
            SEMANTIC_WEIGHT * semantic_score
            + SKILL_WEIGHT * skill_overlap
            + EXPERIENCE_WEIGHT * experience_fit
        )

        candidate["matched_skills"] = matched_skills
        candidate["semantic_score"] = semantic_score
        candidate["skill_overlap"] = skill_overlap
        candidate["experience_fit"] = experience_fit
        candidate["match_score"] = round(
            normalize_score(final_score),
            2,
        )

        scored.append(candidate)

    scored.sort(
        key=lambda candidate: candidate["match_score"],
        reverse=True,
    )

    return scored