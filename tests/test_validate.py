"""Unit tests for evidence validation."""
import pytest
from src.schemas import RequirementAnalysis, Chunk
from src.validate import (
    calibrate_confidence,
    clean_evidence_quote,
    derive_status,
    normalize,
    validate_evidence,
)


def _make_chunk(text: str, chunk_id: str = "chunk_001") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        section="Experience",
        header="Software Engineer, Acme, 2020-2023",
        body=text,
        embed_text=text,
        source="test.pdf",
    )


def _make_analysis(
    evidence: list[str],
    score: float = 1.0,
    confidence: float = 0.9,
    kind: str = "scored",
) -> RequirementAnalysis:
    return RequirementAnalysis(
        requirement_id="R1",
        requirement="Python experience",
        importance="required",
        kind=kind,
        score=score,
        confidence=confidence,
        evidence=evidence,
        reason="test",
        retrieved_chunk_ids=["chunk_001"],
    )


class TestValidateEvidence:
    def test_real_quote_passes(self):
        chunk = _make_chunk("Built Flask APIs for invoice processing.")
        analysis = _make_analysis(["Built Flask APIs for invoice processing."])
        result = validate_evidence(analysis, [chunk])
        assert result.evidence_valid is True
        assert result.score == 1.0

    def test_fabricated_quote_zeroes_score(self):
        chunk = _make_chunk("Built Flask APIs for invoice processing.")
        analysis = _make_analysis(["Built Django APIs for payment processing."])
        result = validate_evidence(analysis, [chunk])
        assert result.evidence_valid is False
        assert result.score == 0.0
        assert result.confidence == 0.0

    def test_case_insensitive(self):
        chunk = _make_chunk("Built FLASK APIS for processing.")
        analysis = _make_analysis(["built flask apis for processing."])
        result = validate_evidence(analysis, [chunk])
        assert result.evidence_valid is True

    def test_whitespace_normalization(self):
        chunk = _make_chunk("Built Flask APIs  for  invoice processing.")
        analysis = _make_analysis(["Built Flask APIs for invoice processing."])
        result = validate_evidence(analysis, [chunk])
        assert result.evidence_valid is True

    def test_positive_score_without_evidence_is_invalid(self):
        chunk = _make_chunk("Built Flask APIs for invoice processing.")
        analysis = _make_analysis([], score=1.0, confidence=0.9)
        result = validate_evidence(analysis, [chunk])
        assert result.evidence_valid is False
        assert result.score == 0.0
        assert result.confidence == 0.0

    def test_zero_score_without_evidence_is_valid(self):
        # A missing requirement is entitled to have nothing to quote.
        chunk = _make_chunk("No AWS experience is stated.")
        analysis = _make_analysis([], score=0.0, confidence=0.9)
        result = validate_evidence(analysis, [chunk])
        assert result.evidence_valid is True
        assert result.score == 0.0

    def test_blank_evidence_is_treated_as_missing(self):
        chunk = _make_chunk("Built Flask APIs.")
        analysis = _make_analysis(["   "], score=0.75, confidence=0.9)
        result = validate_evidence(analysis, [chunk])
        assert result.evidence_valid is False
        assert result.score == 0.0
        assert result.evidence == []

    def test_evidence_with_no_retrieved_chunks_is_invalid(self):
        analysis = _make_analysis(["Built Flask APIs."], score=1.0)
        result = validate_evidence(analysis, [])
        assert result.evidence_valid is False
        assert result.score == 0.0

    def test_blank_quotes_are_stripped_from_valid_evidence(self):
        chunk = _make_chunk("Built Flask APIs for invoice processing.")
        analysis = _make_analysis(["Built Flask APIs", "  "], score=1.0)
        result = validate_evidence(analysis, [chunk])
        assert result.evidence_valid is True
        assert result.evidence == ["Built Flask APIs"]

    def test_multiple_quotes_all_must_pass(self):
        chunk = _make_chunk("Built Flask APIs and Postgres databases.")
        analysis = _make_analysis(["Built Flask APIs", "Used Redis cache"])
        result = validate_evidence(analysis, [chunk])
        assert result.evidence_valid is False
        assert result.score == 0.0

    @pytest.mark.parametrize(
        "quoted",
        [
            '"Built Flask APIs"',
            "'Built Flask APIs'",
            "“Built Flask APIs”",
            "‘Built Flask APIs’",
        ],
    )
    def test_presentation_quote_delimiters_are_removed(self, quoted):
        chunk = _make_chunk("Built Flask APIs")
        analysis = _make_analysis([quoted])

        result = validate_evidence(analysis, [chunk])

        assert result.evidence_valid is True
        assert result.evidence == ["Built Flask APIs"]

    def test_unmatched_quote_delimiter_is_not_removed(self):
        assert clean_evidence_quote('"Built Flask APIs') == '"Built Flask APIs'


class TestDeriveStatus:
    def test_matched(self):
        a = _make_analysis(["evidence"], score=0.75, confidence=0.9)
        a.evidence_valid = True
        result = derive_status(a)
        assert result.status == "matched"

    def test_partially_matched(self):
        a = _make_analysis(["evidence"], score=0.5, confidence=0.8)
        a.evidence_valid = True
        result = derive_status(a)
        assert result.status == "partially_matched"

    def test_missing(self):
        a = _make_analysis([], score=0.0, confidence=1.0)
        a.evidence_valid = True
        result = derive_status(a)
        assert result.status == "missing"

    def test_uncertain_bad_evidence(self):
        a = _make_analysis(["fake quote"], score=1.0, confidence=0.9)
        a.evidence_valid = False
        result = derive_status(a)
        assert result.status == "uncertain"

    def test_uncertain_low_confidence(self):
        a = _make_analysis(["real quote"], score=0.75, confidence=0.5)
        a.evidence_valid = True
        result = derive_status(a)
        assert result.status == "uncertain"


class TestCalibrateConfidence:
    def test_does_not_lift_confidence_for_silent_gate(self):
        a = _make_analysis([], score=0.0, confidence=0.1, kind="gate")
        a.evidence_valid = True

        result = calibrate_confidence(a, full_resume_audit_completed=True)

        assert result.confidence == 0.1
        assert result.gate_status == "unknown"
        assert result.status == "uncertain"

    def test_lifts_verified_match_after_full_resume_audit(self):
        a = _make_analysis(["evidence"], score=1.0, confidence=0.1)
        a.evidence_valid = True

        result = calibrate_confidence(a, full_resume_audit_completed=True)

        assert result.confidence == 0.7
        assert result.status == "matched"

    def test_lifts_verified_absence_after_full_resume_audit(self):
        a = _make_analysis([], score=0.0, confidence=0.0)
        a.evidence_valid = True

        result = calibrate_confidence(a, full_resume_audit_completed=True)

        assert result.confidence == 0.7
        assert result.status == "missing"

    def test_does_not_lift_low_retrieval_confidence(self):
        a = _make_analysis(["evidence"], score=1.0, confidence=0.1)
        a.evidence_valid = True
        a.low_retrieval_confidence = True

        result = calibrate_confidence(a, full_resume_audit_completed=True)

        assert result.confidence == 0.1
        assert result.status == "uncertain"

    def test_does_not_lift_when_reflection_failed(self):
        a = _make_analysis(["evidence"], score=1.0, confidence=0.1)
        a.evidence_valid = True

        result = calibrate_confidence(a, full_resume_audit_completed=False)

        assert result.confidence == 0.1
        assert result.status == "uncertain"

    def test_does_not_lift_invalid_evidence(self):
        a = _make_analysis(["fake"], score=1.0, confidence=0.1)
        a.evidence_valid = False

        result = calibrate_confidence(a, full_resume_audit_completed=True)

        assert result.confidence == 0.1
        assert result.status == "uncertain"
