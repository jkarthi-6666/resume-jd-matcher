"""Unit tests for adversarial reflection correction invariants.

Mocks src.llm.call, the provider-agnostic seam — patching a provider function
would be bypassed under a different LLM_PROVIDER. See tests/conftest.py.
"""
import pytest
from unittest.mock import patch

from src.schemas import RequirementAnalysis, ReflectionResult, Correction
from src.reflector import reflect


RESUME = """Experience
Senior Engineer, Acme, Jan 2020 - Present
Built Python REST APIs. Led Kubernetes migration.

Skills
Python, REST, Kubernetes, Docker
"""

REAL_QUOTE = "Built Python REST APIs."
FABRICATED_QUOTE = "Built Django microservices on AWS Lambda."


def _analysis(
    score: float = 0.5,
    confidence: float = 0.9,
    evidence: list[str] | None = None,
    requirement_id: str = "R1",
) -> RequirementAnalysis:
    return RequirementAnalysis(
        requirement_id=requirement_id,
        requirement="Python experience",
        importance="required",
        score=score,
        confidence=confidence,
        evidence=[REAL_QUOTE] if evidence is None else evidence,
        reason="test",
        retrieved_chunk_ids=["chunk_000"],
        evidence_valid=True,
    )


def _correction(**kwargs) -> Correction:
    defaults = {
        "requirement_id": "R1",
        "old_score": 0.5,
        "new_score": 1.0,
        "direction": "raised",
        "reason": "Evidence was missed by retrieval.",
        "new_evidence": [REAL_QUOTE],
    }
    return Correction(**{**defaults, **kwargs})


def _reflect(corrections: list[Correction], analyses: list[RequirementAnalysis]):
    result = ReflectionResult(
        approved=False, changed_requirements=corrections, review_notes=[]
    )
    with patch("src.llm.call", return_value=result):
        return reflect(analyses, RESUME)


class TestAcceptedCorrections:
    def test_valid_raised_correction_is_applied(self):
        corrected, result = _reflect([_correction()], [_analysis(score=0.5)])
        assert corrected[0].score == 1.0
        assert corrected[0].evidence == [REAL_QUOTE]
        assert len(result.changed_requirements) == 1

    def test_raised_correction_lifts_confidence_to_the_floor(self):
        corrected, _ = _reflect(
            [_correction()], [_analysis(score=0.5, confidence=0.3)]
        )
        assert corrected[0].confidence >= 0.7

    def test_valid_lowered_correction_is_applied(self):
        correction = _correction(
            old_score=0.75, new_score=0.25, direction="lowered", new_evidence=[]
        )
        corrected, result = _reflect([correction], [_analysis(score=0.75)])
        assert corrected[0].score == 0.25
        assert len(result.changed_requirements) == 1

    def test_lowering_to_zero_clears_evidence_and_marks_missing(self):
        correction = _correction(
            old_score=0.75, new_score=0.0, direction="lowered", new_evidence=[]
        )
        corrected, _ = _reflect([correction], [_analysis(score=0.75)])
        assert corrected[0].score == 0.0
        assert corrected[0].evidence == []
        assert corrected[0].status == "missing"

    def test_lowering_gate_to_violation_retains_verified_evidence(self):
        gate = _analysis(score=1.0).model_copy(update={"kind": "gate"})
        correction = _correction(
            old_score=1.0,
            new_score=0.0,
            direction="lowered",
            new_evidence=[REAL_QUOTE],
        )

        corrected, result = _reflect([correction], [gate])

        assert corrected[0].score == 0.0
        assert corrected[0].evidence == [REAL_QUOTE]
        assert corrected[0].gate_status == "violated"
        assert result.changed_requirements == [correction]

    def test_correction_preserves_requirement_order(self):
        analyses = [
            _analysis(score=0.5, requirement_id="R1"),
            _analysis(score=0.5, requirement_id="R2"),
        ]
        corrected, _ = _reflect([_correction(requirement_id="R2")], analyses)
        assert [a.requirement_id for a in corrected] == ["R1", "R2"]
        assert corrected[1].score == 1.0

    def test_prompt_uses_contiguous_text_but_validation_uses_union(self):
        contiguous = "Built Python APIs."
        validation_corpus = "Built Python APIs.\n\nExperience\nBuilt Python APIs."
        correction = _correction(new_evidence=["Experience"])
        result = ReflectionResult(
            approved=False, changed_requirements=[correction], review_notes=[]
        )

        with patch("src.llm.call", return_value=result) as mock_call:
            corrected, reflected = reflect(
                [_analysis(score=0.5)],
                contiguous,
                validation_corpus=validation_corpus,
            )

        prompt = mock_call.call_args.args[0]
        assert validation_corpus not in prompt
        assert prompt.count(contiguous) == 1
        assert corrected[0].score == 1.0
        assert reflected.changed_requirements == [correction]


class TestRejectedCorrections:
    def test_raised_correction_with_fabricated_evidence_is_ignored(self):
        # Regression guard. Checking `score > 0` after validate_evidence would
        # never fire (validation zeroes the score first), so a fabricated
        # correction would be applied as a silent zero instead of ignored.
        correction = _correction(new_evidence=[FABRICATED_QUOTE])
        corrected, result = _reflect([correction], [_analysis(score=0.5)])
        assert corrected[0].score == 0.5
        assert corrected[0].evidence == [REAL_QUOTE]
        assert result.changed_requirements == []
        assert any("could not be verified" in n for n in result.review_notes)
        assert result.rejected_corrections[0].reason == "evidence_substring_miss"

    def test_cross_chunk_quote_is_accepted_against_validation_union(self):
        quote = "Designed distributed systems at scale."
        validation_union = (
            f"Experience\n{quote}\n\n"
            "Experience\nDesigned distributed\n\nExperience\nsystems at scale."
        )
        correction = _correction(new_evidence=[quote])
        result = ReflectionResult(
            approved=False, changed_requirements=[correction], review_notes=[]
        )
        with patch("src.llm.call", return_value=result):
            corrected, reflected = reflect([_analysis(score=0.5)], validation_union)

        assert corrected[0].score == 1.0
        assert reflected.changed_requirements == [correction]

    def test_lowered_correction_with_fabricated_evidence_is_ignored(self):
        correction = _correction(
            old_score=0.75,
            new_score=0.25,
            direction="lowered",
            new_evidence=[FABRICATED_QUOTE],
        )
        corrected, result = _reflect([correction], [_analysis(score=0.75)])
        assert corrected[0].score == 0.75
        assert result.changed_requirements == []

    def test_raised_correction_without_evidence_is_ignored(self):
        correction = _correction(new_evidence=[])
        corrected, result = _reflect([correction], [_analysis(score=0.5)])
        assert corrected[0].score == 0.5
        assert result.changed_requirements == []
        assert any("requires new evidence" in n for n in result.review_notes)

    def test_stale_old_score_is_ignored(self):
        # The critic scored a version of the analysis we no longer hold.
        correction = _correction(old_score=0.25, new_score=0.0, direction="lowered")
        corrected, result = _reflect([correction], [_analysis(score=0.75)])
        assert corrected[0].score == 0.75
        assert result.changed_requirements == []
        assert any("does not match" in n for n in result.review_notes)

    def test_unknown_requirement_id_is_ignored(self):
        correction = _correction(requirement_id="R99")
        corrected, result = _reflect([correction], [_analysis(score=0.5)])
        assert corrected[0].score == 0.5
        assert result.changed_requirements == []
        assert any("unknown requirement ID" in n for n in result.review_notes)

    def test_out_of_range_score_is_ignored(self):
        correction = _correction(new_score=5.0)
        corrected, result = _reflect([correction], [_analysis(score=0.5)])
        assert corrected[0].score == 0.5
        assert result.changed_requirements == []
        assert any("outside [0, 1]" in n for n in result.review_notes)

    def test_raised_direction_that_lowers_score_is_ignored(self):
        correction = _correction(old_score=0.75, new_score=0.25, direction="raised")
        corrected, result = _reflect([correction], [_analysis(score=0.75)])
        assert corrected[0].score == 0.75
        assert result.changed_requirements == []

    def test_lowered_direction_that_raises_score_is_ignored(self):
        correction = _correction(old_score=0.25, new_score=0.75, direction="lowered")
        corrected, result = _reflect([correction], [_analysis(score=0.25)])
        assert corrected[0].score == 0.25
        assert result.changed_requirements == []

    def test_unchanged_score_is_ignored(self):
        correction = _correction(old_score=0.5, new_score=0.5)
        corrected, result = _reflect([correction], [_analysis(score=0.5)])
        assert result.changed_requirements == []
        assert any("unchanged" in n for n in result.review_notes)

    def test_correction_without_reason_is_ignored(self):
        correction = _correction(reason="   ")
        corrected, result = _reflect([correction], [_analysis(score=0.5)])
        assert corrected[0].score == 0.5
        assert result.changed_requirements == []

    def test_one_bad_correction_does_not_discard_the_good_ones(self):
        # Enforcing invariants per-correction rather than in the schema means a
        # single malformed item cannot fail the whole batch.
        analyses = [
            _analysis(score=0.5, requirement_id="R1"),
            _analysis(score=0.5, requirement_id="R2"),
        ]
        corrections = [
            _correction(requirement_id="R1", new_score=5.0),
            _correction(requirement_id="R2", new_score=1.0),
        ]
        corrected, result = _reflect(corrections, analyses)
        assert corrected[0].score == 0.5
        assert corrected[1].score == 1.0
        assert len(result.changed_requirements) == 1


class TestReflectionFailure:
    def test_pre_reflection_report_survives_repeated_failure(self):
        analyses = [_analysis(score=0.5)]
        with patch("src.llm.call", side_effect=RuntimeError("provider down")):
            corrected, result = reflect(analyses, RESUME)
        assert corrected[0].score == 0.5
        assert result.changed_requirements == []
        assert result.review_notes

    def test_all_scores_stay_in_range_after_reflection(self):
        corrections = [
            _correction(requirement_id="R1", new_score=5.0),
            _correction(requirement_id="R2", new_score=-1.0),
        ]
        analyses = [
            _analysis(score=0.5, requirement_id="R1"),
            _analysis(score=0.5, requirement_id="R2"),
        ]
        corrected, _ = _reflect(corrections, analyses)
        assert all(0.0 <= a.score <= 1.0 for a in corrected)
