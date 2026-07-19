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
