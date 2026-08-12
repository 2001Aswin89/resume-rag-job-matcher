from src.scoring import (
    apply_must_have_filter,
    calculate_experience_fit,
    calculate_skill_overlap,
    normalize_score,
    score_candidates,
)


def make_candidate(
    name="Candidate",
    skills=None,
    experience_years=5.0,
    semantic_score=0.5,
):
    return {
        "candidate_name": name,
        "resume_path": f"data/resumes/{name.lower()}.txt",
        "skills": skills or [],
        "experience_years": experience_years,
        "education": [],
        "chunks": [],
        "semantic_scores": [semantic_score],
        "hybrid_scores": [semantic_score],
    }


def test_normalize_score_stays_within_0_and_100():
    assert normalize_score(0.0) == 0.0
    assert normalize_score(0.5) == 50.0
    assert normalize_score(1.0) == 100.0
    assert normalize_score(-1.0) == 0.0
    assert normalize_score(2.0) == 100.0


def test_calculate_skill_overlap():
    overlap, matched = calculate_skill_overlap(
        ["Python", "Docker", "PostgreSQL"],
        ["Python", "FastAPI", "Docker"],
    )

    assert overlap == 2 / 3
    assert matched == ["Python", "Docker"]


def test_calculate_skill_overlap_with_no_required_skills():
    overlap, matched = calculate_skill_overlap(
        ["Python", "Docker"],
        [],
    )

    assert overlap == 0.0
    assert matched == []


def test_experience_fit():
    assert calculate_experience_fit(8.0, 5.0) == 1.0
    assert calculate_experience_fit(5.0, 5.0) == 1.0
    assert calculate_experience_fit(2.5, 5.0) == 0.5
    assert calculate_experience_fit(0.0, None) == 1.0


def test_must_have_filter_excludes_missing_skill():
    candidates = [
        make_candidate(
            name="Good Candidate",
            skills=["Python", "Docker", "AWS"],
            experience_years=7.0,
        ),
        make_candidate(
            name="Missing Docker",
            skills=["Python", "AWS"],
            experience_years=7.0,
        ),
    ]

    filtered = apply_must_have_filter(
        candidates,
        ["Python", "Docker"],
        5.0,
    )

    assert len(filtered) == 1
    assert filtered[0]["candidate_name"] == "Good Candidate"


def test_must_have_filter_excludes_insufficient_experience():
    candidates = [
        make_candidate(
            name="Experienced",
            skills=["Python", "Docker"],
            experience_years=7.0,
        ),
        make_candidate(
            name="Junior",
            skills=["Python", "Docker"],
            experience_years=3.0,
        ),
    ]

    filtered = apply_must_have_filter(
        candidates,
        ["Python", "Docker"],
        5.0,
    )

    assert len(filtered) == 1
    assert filtered[0]["candidate_name"] == "Experienced"


def test_must_have_filter_allows_candidate_when_requirements_are_met():
    candidates = [
        make_candidate(
            name="Qualified",
            skills=["Python", "Docker", "AWS"],
            experience_years=8.0,
        )
    ]

    filtered = apply_must_have_filter(
        candidates,
        ["Python", "Docker"],
        5.0,
    )

    assert len(filtered) == 1
    assert filtered[0]["must_have_skills"] == [
        "Python",
        "Docker",
    ]


def test_score_candidates_is_between_0_and_100():
    candidates = [
        make_candidate(
            name="Strong Candidate",
            skills=["Python", "Docker", "AWS"],
            experience_years=8.0,
            semantic_score=0.9,
        ),
        make_candidate(
            name="Weak Candidate",
            skills=["Python"],
            experience_years=2.0,
            semantic_score=0.1,
        ),
    ]

    ranked = score_candidates(
        candidates,
        ["Python", "Docker", "AWS"],
        5.0,
    )

    assert len(ranked) == 2

    for candidate in ranked:
        assert 0.0 <= candidate["match_score"] <= 100.0


def test_score_candidates_ranks_stronger_candidate_higher():
    candidates = [
        make_candidate(
            name="Weak Candidate",
            skills=["Python"],
            experience_years=2.0,
            semantic_score=0.2,
        ),
        make_candidate(
            name="Strong Candidate",
            skills=["Python", "Docker", "AWS"],
            experience_years=8.0,
            semantic_score=0.9,
        ),
    ]

    ranked = score_candidates(
        candidates,
        ["Python", "Docker", "AWS"],
        5.0,
    )

    assert ranked[0]["candidate_name"] == "Strong Candidate"
    assert (
        ranked[0]["match_score"]
        > ranked[1]["match_score"]
    )