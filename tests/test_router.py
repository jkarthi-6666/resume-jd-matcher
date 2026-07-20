"""Unit tests for the confidence router."""
import pytest
from src.schemas import RequirementAnalysis
from src.router import route
from src.validate import derive_status


def _req(
    importance: str,
    score: float,
    confidence: float,
    evidence_valid: bool | None = True,
    requirement_id: str = "R1",
    requirement: str = "Test requirement",
    kind: str = "scored",
    evidence: list[str] | None = None,
) -> RequirementAnalysis:
    """Distinct IDs and names by default would hide ordering and mapping bugs,
    so callers testing multi-requirement routing should pass them explicitly."""
    analysis = RequirementAnalysis(
        requirement_id=requirement_id,
        requirement=requirement,
        importance=importance,
        kind=kind,
        score=score,
        confidence=confidence,
        evidence=evidence if evidence is not None else (["some evidence"] if score > 0 else []),
        reason="test",
        retrieved_chunk_ids=[],
        evidence_valid=evidence_valid,
    )
    return derive_status(analysis)


class TestRoute:
    def test_accept(self):
        analyses = [
            _req("required", 1.0, 0.9),
            _req("preferred", 0.5, 0.8),
        ]
        verdict, reason = route(analyses)
        assert verdict == "accept"
        assert reason is None

    def test_reject_all_zero(self):
        analyses = [
            _req("required", 0.0, 1.0),
            _req("required", 0.0, 1.0),
        ]
        verdict, reason = route(analyses)
        assert verdict == "reject"

    def test_needs_review_bad_evidence(self):
        analyses = [
            _req("required", 1.0, 0.9, evidence_valid=False),
        ]
        verdict, reason = route(analyses)
        assert verdict == "needs_review"
        assert "could not be verified" in reason.lower()

    def test_needs_review_low_confidence(self):
        analyses = [
            _req("required", 1.0, 0.5),
        ]
        verdict, reason = route(analyses, confidence_floor=0.7)
        assert verdict == "needs_review"

    def test_preferred_only_no_required(self):
        # No required requirements → accept (nothing to fail)
        analyses = [
            _req("preferred", 0.0, 1.0),
        ]
        verdict, reason = route(analyses)
        assert verdict == "accept"

    def test_custom_confidence_floor(self):
        analyses = [_req("required", 1.0, 0.65)]
        verdict_strict, _ = route(analyses, confidence_floor=0.7)
        verdict_loose, _  = route(analyses, confidence_floor=0.6)
        assert verdict_strict == "needs_review"
        assert verdict_loose  == "accept"


class TestEligibilityGates:
    def test_unknown_gate_routes_to_review(self):
        gate = _req(
            "required", 0.0, 0.95, kind="gate",
            requirement="Must be authorized to work in the US without sponsorship",
        )
        assert gate.gate_status == "unknown"
        verdict, reason = route([gate])
        assert verdict == "needs_review"
        assert "authorized to work" in reason
        assert "unresolved" in reason

    def test_violated_gate_routes_to_review_by_default(self):
        gate = _req(
            "required", 0.0, 0.95, kind="gate",
            requirement="Must be authorized to work in the US without sponsorship",
            evidence=["I require employer visa sponsorship."],
        )
        assert gate.gate_status == "violated"
        verdict, reason = route([gate], reject_violated_gates=False)
        assert verdict == "needs_review"
        assert "violated" in reason

    def test_violated_gate_can_reject_under_policy_flag(self):
        gate = _req(
            "required", 0.0, 0.95, kind="gate",
            evidence=["I require employer visa sponsorship."],
        )
        verdict, _ = route([gate], reject_violated_gates=True)
        assert verdict == "reject"

    def test_satisfied_gate_preserves_scored_requirement_precedence(self):
        gate = _req("required", 1.0, 0.95, kind="gate")
        missing = _req("required", 0.0, 0.95, requirement="AWS")
        verdict, reason = route([gate, missing])
        assert verdict == "reject"
        assert "AWS" in reason


class TestMissingRequired:
    def test_reject_when_one_of_two_required_is_missing(self):
        # The whole point of the fix: meeting Python does not excuse missing AWS.
        analyses = [
            _req("required", 1.0, 0.9, requirement_id="R1", requirement="Python"),
            _req("required", 0.0, 0.9, requirement_id="R2", requirement="AWS"),
        ]
        verdict, reason = route(analyses)
        assert verdict == "reject"
        assert "AWS" in reason

    def test_reject_names_the_weakest_required_item(self):
        analyses = [
            _req("required", 0.1, 0.9, requirement_id="R1", requirement="Kubernetes"),
            _req("required", 0.0, 0.9, requirement_id="R2", requirement="AWS"),
        ]
        verdict, reason = route(analyses)
        assert verdict == "reject"
        assert "AWS" in reason

    def test_partial_required_routes_to_review(self):
        analyses = [_req("required", 0.5, 0.9)]
        verdict, reason = route(analyses)
        assert verdict == "needs_review"
        assert "partially supported" in reason

    def test_all_required_at_accept_threshold_are_accepted(self):
        analyses = [
            _req("required", 0.75, 0.9, requirement_id="R1"),
            _req("required", 1.0, 0.9, requirement_id="R2"),
        ]
        verdict, reason = route(analyses)
        assert verdict == "accept"
        assert reason is None

    def test_preferred_items_never_cause_rejection(self):
        analyses = [
            _req("required", 1.0, 0.9, requirement_id="R1"),
            _req("preferred", 0.0, 0.9, requirement_id="R2"),
        ]
        verdict, _ = route(analyses)
        assert verdict == "accept"

    def test_unassessable_required_reviews_rather_than_rejects(self):
        # A requirement the system could not assess is not proof of absence,
        # so low confidence must win over the missing-required reject rule.
        analyses = [_req("required", 0.0, 0.2, requirement_id="R1")]
        verdict, _ = route(analyses)
        assert verdict == "needs_review"

    def test_unverifiable_evidence_reviews_rather_than_rejects(self):
        analyses = [_req("required", 0.0, 0.9, evidence_valid=False)]
        verdict, _ = route(analyses)
        assert verdict == "needs_review"

    def test_custom_accept_threshold(self):
        analyses = [_req("required", 0.8, 0.9)]
        verdict_strict, _ = route(analyses, required_accept_score=0.9)
        verdict_loose, _  = route(analyses, required_accept_score=0.75)
        assert verdict_strict == "needs_review"
        assert verdict_loose  == "accept"
