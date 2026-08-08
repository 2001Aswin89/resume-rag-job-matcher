from pathlib import Path

from src.hybrid_search import HybridSearcher
from src.llm_reasoning import generate_reasoning
from src.scoring import (
    aggregate_candidates,
    apply_must_have_filter,
    extract_must_have_skills,
    extract_required_experience,
    extract_required_skills,
    score_candidates,
)


TOP_K_RETRIEVAL = 20
TOP_K_RESULTS = 10


def load_job_description(path: str) -> str:
    """
    Load a job description from a text file.
    """
    return Path(path).read_text(
        encoding="utf-8"
    )


def match_job(
    job_description: str,
    top_k: int = TOP_K_RESULTS,
) -> dict:
    """
    Match a job description against the indexed resumes.

    Returns a structured result containing the ranked
    candidate matches.
    """
    searcher = HybridSearcher()

    # ---------------------------------------------------------
    # 1. Retrieve relevant resume chunks
    # ---------------------------------------------------------

    search_results = searcher.search(
        job_description,
        top_k=TOP_K_RETRIEVAL,
    )

    # ---------------------------------------------------------
    # 2. Extract job requirements
    # ---------------------------------------------------------

    required_skills = extract_required_skills(
        job_description
    )

    must_have_skills = extract_must_have_skills(
        job_description,
        required_skills,
    )

    required_years = extract_required_experience(
        job_description
    )

    # ---------------------------------------------------------
    # 3. Aggregate chunks by candidate
    # ---------------------------------------------------------

    candidates = aggregate_candidates(
        search_results
    )

    # ---------------------------------------------------------
    # 4. Apply hard requirements
    # ---------------------------------------------------------

    candidates = apply_must_have_filter(
        candidates,
        must_have_skills,
        required_years,
    )

    # ---------------------------------------------------------
    # 5. Score and rank candidates
    # ---------------------------------------------------------

    ranked_candidates = score_candidates(
        candidates,
        required_skills,
        required_years,
    )

    # ---------------------------------------------------------
    # 6. Limit final results
    # ---------------------------------------------------------

    ranked_candidates = ranked_candidates[:top_k]

    # ---------------------------------------------------------
    # 7. Add reasoning
    # ---------------------------------------------------------

    matches = []

    for candidate in ranked_candidates:
        reasoning = generate_reasoning(
            job_description,
            candidate,
        )

        relevant_excerpts = [
            {
                "section": chunk["section"],
                "text": chunk["text"],
            }
            for chunk in candidate.get(
                "chunks",
                [],
            )[:3]
        ]

        matches.append(
            {
                "candidate_name": candidate[
                    "candidate_name"
                ],
                "resume_path": candidate[
                    "resume_path"
                ],
                "match_score": candidate[
                    "match_score"
                ],
                "matched_skills": candidate[
                    "matched_skills"
                ],
                "experience_years": candidate[
                    "experience_years"
                ],
                "relevant_excerpts": relevant_excerpts,
                "reasoning": reasoning,
            }
        )

    # ---------------------------------------------------------
    # 8. Return structured result
    # ---------------------------------------------------------

    return {
        "job_description": job_description,
        "required_skills": required_skills,
        "must_have_skills": must_have_skills,
        "required_experience_years": required_years,
        "total_candidates": len(matches),
        "top_matches": matches,
    }


def print_results(result: dict) -> None:
    """
    Print matching results in a demo-friendly format.
    """
    print()
    print("=" * 70)
    print("RESUME JOB MATCHING RESULTS")
    print("=" * 70)

    print()
    print("Required skills:")
    print(", ".join(result["required_skills"]))

    print()
    print("Must-have skills:")
    print(", ".join(result["must_have_skills"]))

    print()
    print(
        "Required experience:",
        result["required_experience_years"],
        "years",
    )

    print()
    print(
        "Candidates matched:",
        result["total_candidates"],
    )

    print()
    print("-" * 70)

    for index, match in enumerate(
        result["top_matches"],
        start=1,
    ):
        print()
        print(
            f"#{index} {match['candidate_name']}"
        )

        print(
            f"Match Score: "
            f"{match['match_score']}"
        )

        print(
            f"Experience: "
            f"{match['experience_years']} years"
        )

        print(
            "Matched Skills: "
            + ", ".join(
                match["matched_skills"]
            )
        )

        print(
            "Reasoning: "
            + match["reasoning"]
        )

        if match["relevant_excerpts"]:
            print("Relevant sections:")

            for excerpt in match[
                "relevant_excerpts"
            ]:
                print(
                    f"  - {excerpt['section']}"
                )

        print("-" * 70)


def main() -> None:
    """
    Command-line entry point.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Match resumes against a job description."
    )

    parser.add_argument(
        "--jd",
        required=True,
        help="Path to the job description text file.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K_RESULTS,
        help="Number of candidates to return.",
    )

    args = parser.parse_args()

    job_description = load_job_description(
        args.jd
    )

    result = match_job(
        job_description,
        top_k=args.top_k,
    )

    print_results(result)


if __name__ == "__main__":
    main()
