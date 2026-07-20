"""Weighted score computation. Python computes the score; the model never does."""
from src.schemas import RequirementAnalysis

_WEIGHTS = {"required": 2, "preferred": 1, "unknown": 1}


def weighted_score(analyses: list[RequirementAnalysis]) -> float:
    """Return score in [0, 100]."""
    earned = 0.0
    possible = 0.0
    for a in analyses:
        if a.kind == "gate":
            continue
        if not 0.0 <= a.score <= 1.0:
            raise ValueError(
                f"Invalid score for {a.requirement_id}: {a.score}"
            )
        w = _WEIGHTS.get(a.importance, 1)
        earned   += a.score * w
        possible += 1.0   * w
    if possible == 0:
        return 0.0
    result = round((earned / possible) * 100, 1)
    return min(100.0, max(0.0, result))
