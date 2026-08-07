import re


SECTION_HEADERS = [
    "Summary",
    "Professional Summary",
    "Objective",
    "Skills",
    "Technical Skills",
    "Experience",
    "Professional Experience",
    "Work Experience",
    "Education",
    "Projects",
    "Certifications",
    "Achievements",
    "Awards",
    "Languages",
    "Interests",
    "Publications",
]

HEADER_PATTERN = re.compile(
    rf"^\s*({'|'.join(re.escape(h) for h in SECTION_HEADERS)})\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def chunk_resume(text: str) -> list[dict]:
    """
    Split a resume into logical sections.

    Returns:
        [
            {
                "section": "...",
                "text": "..."
            }
        ]
    """

    matches = list(HEADER_PATTERN.finditer(text))

    if not matches:
        return [
            {
                "section": "General",
                "text": text.strip(),
            }
        ]

    chunks = []

    for index, match in enumerate(matches):
        section = match.group(1).strip()

        start = match.end()

        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(text)
        )

        section_text = text[start:end].strip()

        if section_text:
            chunks.append(
                {
                    "section": section,
                    "text": section_text,
                }
            )

    return chunks