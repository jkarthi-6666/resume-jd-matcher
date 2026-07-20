"""Adversarial reflection. One pass, no scorer reasoning, full resume access."""
import json
import math
import pathlib
from src.schemas import (
    Chunk, RequirementAnalysis, ReflectionResult, Correction,
    CorrectionRejection, CorrectionRejectionReason,
)
from src.validate import validate_evidence, derive_status
from src import llm, config

_PROMPT = (pathlib.Path(__file__).parent.parent / "prompts" / "reflection.txt").read_text()


def reflect(
    analyses: list[RequirementAnalysis],
    resume_text: str,
    model: str | None = None,
    validation_corpus: str | None = None,
) -> tuple[list[RequirementAnalysis], ReflectionResult]:
    """
    Run adversarial reflection. Returns (corrected_analyses, reflection_result).
    One pass; if Pydantic validation fails, retry once, then ship pre-reflection.
    """
    model = model or config.REFLECTOR_MODEL

    # Build the slim view the critic receives (no reason field)
    reqs_for_critic = [
        {
            "requirement_id": a.requirement_id,
            "requirement": a.requirement,
            "importance": a.importance,
            "kind": a.kind,
            "score": a.score,
            "evidence": a.evidence,
            "low_retrieval_confidence": a.low_retrieval_confidence,
        }
        for a in analyses
    ]

    prompt = _PROMPT.format(
        resume_text=resume_text,
        requirements_json=json.dumps(reqs_for_critic, indent=2),
    )

    result: ReflectionResult | None = None
    for attempt in range(2):
        try:
            result = llm.call(prompt, model, ReflectionResult, temperature=0.0)
            break
        except Exception:
            if attempt == 1:
                # Ship pre-reflection report with warning
                null_result = ReflectionResult(
                    approved=True,
                    changed_requirements=[],
                    review_notes=["Reflection failed after retry — pre-reflection report used."],
                    completed=False,
                )
                return analyses, null_result

    if result is None:
        null_result = ReflectionResult(
            approved=True,
            changed_requirements=[],
            review_notes=["Reflection produced no result."],
            completed=False,
        )
        return analyses, null_result

    # Apply corrections
    analysis_map = {a.requirement_id: a for a in analyses}
    accepted_corrections: list[Correction] = []

    evidence_corpus = validation_corpus or resume_text
    resume_as_chunk = Chunk(
        chunk_id="full_resume",
        section="Full",
        header="Full resume",
        body=evidence_corpus,
        embed_text=evidence_corpus,
        source="resume.pdf",
    )

    for correction in result.changed_requirements:
        rid = correction.requirement_id

        def _ignore(code: CorrectionRejectionReason, detail: str) -> None:
            result.rejected_corrections.append(CorrectionRejection(
                requirement_id=rid,
                reason=code,
                detail=detail,
            ))
            result.review_notes.append(f"Correction for {rid} ignored: {detail}")

        original = analysis_map.get(rid)
        if original is None:
            _ignore("unknown_requirement", f"unknown requirement ID {rid!r}.")
            continue

        if not _is_valid_correction(correction, original, _ignore):
            continue

        candidate = original.model_copy(deep=True)
        candidate.score = correction.new_score

        if correction.new_evidence:
            candidate.evidence = correction.new_evidence
        elif correction.direction == "raised" or original.kind == "gate":
            _ignore("missing_new_evidence", "this correction requires new evidence.")
            continue

        # A zeroed scored requirement claims nothing. A zeroed gate claims an
        # explicit violation, so retain and validate the critic's evidence.
        if candidate.score == 0.0 and candidate.kind == "scored":
            candidate.evidence = []

        # Re-validate against the full resume, not just the retrieved chunks —
        # the critic sees the whole document and may cite outside them.
        candidate = validate_evidence(candidate, [resume_as_chunk])
        # Check evidence_valid directly. validate_evidence zeroes the score when
        # evidence fails, so a `candidate.score > 0` guard here would never fire
        # and fabricated evidence would be applied as a silent zero instead of
        # being rejected outright.
        if candidate.evidence_valid is not True:
            _ignore(
                "evidence_substring_miss",
                "supporting evidence could not be verified against the resume.",
            )
            continue

        if correction.direction == "raised":
            candidate.confidence = max(candidate.confidence, config.CONFIDENCE_FLOOR)

        derive_status(candidate)
        analysis_map[rid] = candidate
        accepted_corrections.append(correction)

    result.changed_requirements = accepted_corrections
    corrected = list(analysis_map.values())
    return corrected, result


def _is_valid_correction(
    correction: Correction,
    original: RequirementAnalysis,
    ignore,
) -> bool:
    """Invariants enforced here rather than in the schema, so one malformed
    correction cannot fail validation for the whole batch."""
    if not 0.0 <= correction.new_score <= 1.0:
        ignore(
            "invalid_score_bounds",
            f"new_score {correction.new_score} is outside [0, 1].",
        )
        return False

    # A correction quoting a score we never assigned is stale or fabricated;
    # applying it would overwrite the real score with an unrelated judgment.
    if not math.isclose(
        correction.old_score, original.score, rel_tol=0.0, abs_tol=1e-6
    ):
        ignore(
            "stale_old_score",
            f"old_score {correction.old_score} does not match the current "
            f"score {original.score}."
        )
        return False

    if not correction.reason.strip():
        ignore("missing_reason", "no reason was given.")
        return False

    if correction.new_score == correction.old_score:
        ignore("unchanged_score", "the score is unchanged.")
        return False

    if correction.direction == "raised" and correction.new_score < correction.old_score:
        ignore("direction_mismatch", "direction is 'raised' but the score decreases.")
        return False

    if correction.direction == "lowered" and correction.new_score > correction.old_score:
        ignore("direction_mismatch", "direction is 'lowered' but the score increases.")
        return False

    return True
