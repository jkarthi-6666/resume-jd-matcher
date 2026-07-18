"""Unit tests for weighted score computation."""
import pytest
from src.schemas import RequirementAnalysis
from src.calculator import weighted_score


def _req(score: float, importance: str) -> RequirementAnalysis:
    return RequirementAnalysis(
        requirement_id="R1",
        requirement="test",
        importance=importance,
        score=score,
        confidence=0.9,
        evidence=[],
        reason="",
        retrieved_chunk_ids=[],
    )


class TestWeightedScore:
    def test_all_required_perfect(self):
        analyses = [
            _req(1.0, "required"),
            _req(1.0, "required"),
        ]
        assert weighted_score(analyses) == 100.0

    def test_required_outweighs_preferred(self):
        # required=2, preferred=1
        # Python (required) 1.0 × 2 = 2.0
        # REST APIs (required) 1.0 × 2 = 2.0
        # Kubernetes (preferred) 0.0 × 1 = 0.0
        # AWS (preferred) 0.5 × 1 = 0.5
        # earned=4.5, possible=6.0 → 75
        analyses = [
            _req(1.0, "required"),
            _req(1.0, "required"),
            _req(0.0, "preferred"),
            _req(0.5, "preferred"),
        ]
        assert weighted_score(analyses) == 75.0

    def test_zero_analyses(self):
        assert weighted_score([]) == 0.0

    def test_all_zeros(self):
        analyses = [_req(0.0, "required"), _req(0.0, "preferred")]
        assert weighted_score(analyses) == 0.0

    def test_mixed(self):
        analyses = [
            _req(1.0, "required"),   # 2.0
            _req(0.0, "required"),   # 0.0
            _req(1.0, "preferred"),  # 1.0
        ]
        # earned=3.0, possible=5.0 → 60.0
        assert weighted_score(analyses) == 60.0

    def test_never_exceeds_one_hundred(self):
        analyses = [_req(1.0, "required"), _req(1.0, "preferred")]
        assert weighted_score(analyses) <= 100.0


class TestScoreInvariant:
    def test_out_of_range_score_raises(self):
        # Field(ge/le) only guards __init__, and RequirementAnalysis is mutated
        # after construction, so the calculator re-checks rather than trusting it.
        analysis = _req(1.0, "required")
        object.__setattr__(analysis, "__dict__", {**analysis.__dict__, "score": 5.0})
        with pytest.raises(ValueError, match="Invalid score"):
            weighted_score([analysis])
