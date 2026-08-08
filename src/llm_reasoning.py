import os

from dotenv import load_dotenv
from groq import Groq


load_dotenv()


MODEL_NAME = "llama-3.1-8b-instant"


def _template_reasoning(candidate: dict) -> str:
    """
    Generate deterministic reasoning when Groq is unavailable.
    """
    skills = candidate.get("matched_skills", [])
    sections = []

    for chunk in candidate.get("chunks", []):
        section = chunk.get("section")

        if section and section not in sections:
            sections.append(section)

    if skills:
        skill_text = ", ".join(skills)
    else:
        skill_text = "relevant skills"

    if sections:
        section_text = ", ".join(sections)
    else:
        section_text = "relevant resume sections"

    return (
        f"Matched on {skill_text}; "
        f"relevant experience found in {section_text}."
    )


def generate_reasoning(
    job_description: str,
    candidate: dict,
) -> str:
    """
    Generate a short explanation for a candidate match.

    Groq is used when GROQ_API_KEY is available.
    Any API failure falls back to deterministic reasoning.
    """
    fallback = _template_reasoning(candidate)

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return fallback

    try:
        client = Groq(api_key=api_key)

        skills = ", ".join(
            candidate.get("matched_skills", [])
        )

        excerpts = "\n".join(
            chunk.get("text", "")
            for chunk in candidate.get("chunks", [])[:3]
        )

        prompt = f"""
You are evaluating a candidate against a job description.

Job Description:
{job_description}

Candidate:
{candidate.get("candidate_name", "")}

Experience:
{candidate.get("experience_years", 0)} years

Matched Skills:
{skills}

Relevant Resume Excerpts:
{excerpts}

Write a concise 1-2 sentence explanation of why this candidate
matches the job description. Mention the strongest matching skills
or experience. Do not invent information.
"""

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0,
            max_tokens=120,
        )

        reasoning = response.choices[0].message.content

        if reasoning:
            return reasoning.strip()

    except Exception:
        return fallback

    return fallback