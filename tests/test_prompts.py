from pathlib import Path


PROMPTS = Path(__file__).parent.parent / "prompts"


def test_scorer_prompt_does_not_anchor_confidence_at_zero():
    prompt = (PROMPTS / "scorer.txt").read_text()

    assert '"confidence": 0.0' not in prompt
    assert "Choose confidence independently" in prompt


def test_reflector_does_not_confuse_importance_with_evidence_score():
    prompt = (PROMPTS / "reflection.txt").read_text()

    assert "A preferred requirement can be\nfully supported and score 1.0" in prompt
    assert "never lower a score merely because the requirement is preferred" in prompt


def test_scorer_and_reflector_agree_that_adjacent_evidence_is_weak_not_absent():
    scorer = (PROMPTS / "scorer.txt").read_text()
    reflector = (PROMPTS / "reflection.txt").read_text()

    assert "keyword listed without demonstrated use is weak evidence" in scorer
    assert "Academic or personal-project evidence" in scorer
    assert "Do not lower 0.25 to 0.0" in reflector


def test_reflector_does_not_invent_professional_constraint_from_projects_heading():
    prompt = (PROMPTS / "reflection.txt").read_text()

    assert "do not\nassume professional employment is required" in prompt
    assert '"Projects" heading does not by itself mean academic' in prompt


def test_planner_preserves_requirement_strength():
    prompt = (PROMPTS / "planner.txt").read_text()

    assert "Preserve the exact strength, seniority, duration, and depth" in prompt
    assert 'Do not rewrite "experience with" as "proficiency"' in prompt


def test_planner_allows_unknown_category_as_a_last_resort():
    prompt = (PROMPTS / "planner.txt").read_text()

    assert "responsibility, unknown" in prompt
    assert "Use unknown only when" in prompt
    assert "none of the other categories applies" in prompt


def test_planner_separates_qualifications_from_recruitment_logistics():
    prompt = (PROMPTS / "planner.txt").read_text()

    assert "notice period" in prompt
    assert "interview or walk-in" in prompt
    assert "Do not turn these into\n   candidate requirements" in prompt


def test_planner_preserves_alternatives_and_parenthetical_examples():
    prompt = (PROMPTS / "planner.txt").read_text()

    assert "Preserve AND/OR" in prompt
    assert "parenthetical examples" in prompt
    assert "not three separately mandatory products" in prompt


def test_planner_requires_grounded_spans_gates_and_bounded_query_terms():
    prompt = (PROMPTS / "planner.txt").read_text()

    assert 'kind "gate"' in prompt
    assert "exact substring, not a paraphrase" in prompt
    assert "up to 5 query_terms" in prompt
    assert "never related or adjacent" in prompt


def test_reflector_does_not_invent_advanced_depth_requirements():
    prompt = (PROMPTS / "reflection.txt").read_text()

    assert "do not demand advanced depth such as schema design" in prompt
    assert 'fully support a plain\n"experience with" requirement' in prompt
