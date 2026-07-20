"""Confidence and requirement routing: accept / needs_review / reject.

Policy, in order of precedence:

    unresolved eligibility gate                -> needs_review
    violated gate (default policy)              -> needs_review
    unverifiable evidence on a required item  -> needs_review
    low confidence on a required item         -> needs_review
    any required item below PARTIAL_MATCH_SCORE  -> reject
    any required item below REQUIRED_ACCEPT_SCORE -> needs_review
    every required item at REQUIRED_ACCEPT_SCORE or above -> accept

The two review rules come first on purpose: a requirement the system could not
assess is not evidence that the candidate lacks it, so it must not reject.
"""
from src.schemas import RequirementAnalysis, Verdict
from src import config


def route(
    analyses: list[RequirementAnalysis],
    confidence_floor: float | None = None,
    required_accept_score: float | None = None,
    partial_match_score: float | None = None,
    reject_violated_gates: bool | None = None,
) -> tuple[Verdict, str | None]:
    floor = (
        confidence_floor
        if confidence_floor is not None
        else config.CONFIDENCE_FLOOR
    )
    accept_score = (
        required_accept_score
        if required_accept_score is not None
        else config.REQUIRED_ACCEPT_SCORE
    )
    missing_score = (
        partial_match_score
        if partial_match_score is not None
        else config.PARTIAL_MATCH_SCORE
    )
    reject_gates = (
        reject_violated_gates
        if reject_violated_gates is not None
        else config.REJECT_VIOLATED_GATES
    )

    gates = [a for a in analyses if a.kind == "gate"]
    unresolved = [a for a in gates if a.gate_status != "satisfied"]
    if unresolved:
        gate = unresolved[0]
        if gate.gate_status == "violated" and reject_gates:
            return "reject", f"Eligibility constraint is not met: {gate.requirement}."
        state = "violated" if gate.gate_status == "violated" else "unresolved"
        return "needs_review", (
            f"Eligibility constraint is {state}: {gate.requirement}. "
            "Confirm this constraint with the candidate."
        )

    required = [
        a for a in analyses if a.kind == "scored" and a.importance == "required"
    ]

    # Unverifiable evidence means the automated decision is unreliable.
    bad_evidence = [a for a in required if a.evidence_valid is False]
    if bad_evidence:
        r = bad_evidence[0]
        return "needs_review", (
            f"Evidence for '{r.requirement}' could not be verified against the resume text."
        )

    # Low model confidence should not produce an automatic decision either way.
    unsure = [a for a in required if a.confidence < floor]
    if unsure:
        r = min(unsure, key=lambda a: a.confidence)
        return "needs_review", (
            f"Could not confidently assess '{r.requirement}'. {r.reason}"
        )

    # One confidently missing required qualification is enough to reject.
    missing = [a for a in required if a.score < missing_score]
    if missing:
        r = min(missing, key=lambda a: a.score)
        return "reject", f"Missing required qualification: {r.requirement}."

    # Partial evidence for a required qualification is a human judgment call.
    partial = [a for a in required if a.score < accept_score]
    if partial:
        r = min(partial, key=lambda a: a.score)
        return "needs_review", (
            f"Required qualification is only partially supported: {r.requirement}."
        )

    return "accept", None
