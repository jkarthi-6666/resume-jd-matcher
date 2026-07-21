"""Unit tests for the offline confidence-floor sweep.

The sweep replays saved analyses through the router and makes no provider
calls, so these tests need no mocking seam.
"""
import json

from evaluation.sweep_threshold import (
    load_sweep_cases,
    measure_floor,
    sweep,
)


def _analysis(score: float = 1.0, confidence: float = 0.9) -> dict:
    return {
        "requirement_id": "R1",
        "requirement": "Python experience",
        "importance": "required",
        "kind": "scored",
        "score": score,
        "confidence": confidence,
        "evidence": ["Built Python REST APIs."],
        "reason": "test",
        "retrieved_chunk_ids": ["chunk_000"],
        "evidence_valid": True,
    }


def _case(
    case_id: str,
    verdict: str,
    expected: str | None,
    analyses: list[dict] | None = None,
) -> dict:
    return {
        "case_id": case_id,
        "status": "completed",
        "expected_verdict": expected,
        "report": {
            "verdict": verdict,
            "all_requirements": [] if analyses is None else analyses,
        },
    }


class TestThresholdIndependentCases:
    def test_case_with_no_requirements_keeps_its_recorded_verdict(self):
        # Regression guard. An unassessable document short-circuits before any
        # requirement is scored, so route([]) would fall through to `accept` and
        # report an unreadable resume as a hire.
        payload = {
            "cases": [_case("case_007", "needs_review", "needs_review")]
        }

        cases = load_sweep_cases(payload)

        assert cases[0].threshold_independent is True
        assert cases[0].verdict_at(0.7) == "needs_review"
        assert cases[0].verdict_at(0.5) == "needs_review"

    def test_empty_requirements_case_is_never_counted_as_a_false_accept(self):
        payload = {"cases": [_case("case_007", "needs_review", "needs_review")]}

        metrics = measure_floor(load_sweep_cases(payload), 0.7)

        assert metrics.false_accepts == 0
        assert metrics.escalated == 1
        assert metrics.correct == 1

    def test_short_circuited_case_counts_toward_escalation_at_every_floor(self):
        payload = {"cases": [_case("case_007", "needs_review", "needs_review")]}
        cases = load_sweep_cases(payload)

        for floor in (0.5, 0.7, 0.9):
            assert measure_floor(cases, floor).escalation_rate == 1.0

    def test_scored_case_is_not_treated_as_threshold_independent(self):
        payload = {
            "cases": [_case("case_001", "accept", "accept", [_analysis()])]
        }

        assert load_sweep_cases(payload)[0].threshold_independent is False


class TestFloorMetrics:
    def test_confidence_floor_changes_verdict_and_accuracy(self):
        # Confidence 0.8 clears a 0.7 floor but not a 0.9 floor.
        payload = {
            "cases": [
                _case(
                    "case_001",
                    "accept",
                    "accept",
                    [_analysis(score=1.0, confidence=0.8)],
                )
            ]
        }
        cases = load_sweep_cases(payload)

        low = measure_floor(cases, 0.7)
        high = measure_floor(cases, 0.9)

        assert low.correct == 1 and low.escalated == 0
        assert high.correct == 0 and high.escalated == 1
        assert high.accuracy == 0.0

    def test_false_accept_and_false_reject_are_counted_separately(self):
        payload = {
            "cases": [
                # Fully supported but labeled reject -> false accept.
                _case("a", "accept", "reject", [_analysis(score=1.0)]),
                # Confidently absent but labeled accept -> false reject.
                _case("b", "reject", "accept", [_analysis(score=0.0)]),
            ]
        }

        metrics = measure_floor(load_sweep_cases(payload), 0.7)

        assert metrics.false_accepts == 1
        assert metrics.false_rejects == 1
        assert metrics.correct == 0
        assert metrics.labeled == 2

    def test_unlabeled_cases_are_excluded_from_accuracy_but_not_escalation(self):
        payload = {
            "cases": [
                _case("a", "needs_review", None, [_analysis(confidence=0.1)]),
                _case("b", "accept", "accept", [_analysis()]),
            ]
        }

        metrics = measure_floor(load_sweep_cases(payload), 0.7)

        assert metrics.labeled == 1
        assert metrics.completed == 2
        assert metrics.escalated == 1
        assert metrics.accuracy == 1.0

    def test_rates_are_none_rather_than_zero_when_nothing_to_divide_by(self):
        metrics = measure_floor([], 0.7)

        assert metrics.accuracy is None
        assert metrics.escalation_rate is None


class TestCaseSelection:
    def test_skipped_and_failed_cases_are_excluded(self):
        payload = {
            "cases": [
                _case("ok", "accept", "accept", [_analysis()]),
                {"case_id": "bad", "status": "failed", "expected_verdict": "accept"},
                {"case_id": "gone", "status": "skipped", "expected_verdict": "accept"},
            ]
        }

        assert [case.case_id for case in load_sweep_cases(payload)] == ["ok"]

    def test_legacy_bare_list_of_reports_is_still_accepted(self):
        payload = [{"verdict": "accept", "all_requirements": [_analysis()]}]

        cases = load_sweep_cases(payload)

        assert len(cases) == 1
        assert cases[0].expected_verdict is None


class TestReport:
    def test_report_shows_numerators_and_denominators(self, tmp_path):
        path = tmp_path / "results.json"
        path.write_text(
            json.dumps(
                {
                    "cases": [
                        _case("case_001", "accept", "accept", [_analysis()]),
                        _case("case_007", "needs_review", "needs_review"),
                    ]
                }
            ),
            encoding="utf-8",
        )

        report = sweep(str(path), floors=[0.7])

        assert "(2/2)" in report  # accuracy numerator/denominator
        assert "(1/2)" in report  # escalation numerator/denominator
        assert "case_007" in report
        assert "threshold-independent" in report
