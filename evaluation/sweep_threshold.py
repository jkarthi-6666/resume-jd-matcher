"""Sweep the confidence floor and report the accuracy cost of each choice.

Reads a saved evaluation run and re-routes it at a range of confidence floors.
This is offline: it replays stored requirement analyses through the router and
makes no model or embedding calls.

Escalation rate alone cannot justify a floor, because escalating everything
scores a perfect zero on both error types. Each floor is therefore reported with
the verdict accuracy and the false accepts and false rejects it buys.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import dataclass
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.router import route  # noqa: E402
from src.schemas import RequirementAnalysis  # noqa: E402

DEFAULT_FLOORS = [round(step * 0.1, 1) for step in range(5, 10)]


@dataclass(frozen=True)
class SweepCase:
    """One completed case, replayable at any confidence floor."""

    case_id: str
    analyses: list[RequirementAnalysis]
    recorded_verdict: str
    expected_verdict: str | None

    @property
    def threshold_independent(self) -> bool:
        """True when no confidence floor can change this case's verdict.

        The pipeline short-circuits an unassessable document before any
        requirement is scored and returns its verdict directly, so the saved
        analyses are empty. Re-routing that empty list asks the router a
        question it was never asked at runtime, and it answers ``accept`` —
        reporting an unreadable resume as a hire, which is the most dangerous
        output this tool could produce.
        """
        return not self.analyses

    def verdict_at(self, floor: float) -> str:
        if self.threshold_independent:
            return self.recorded_verdict
        verdict, _ = route(self.analyses, confidence_floor=floor)
        return verdict


@dataclass(frozen=True)
class FloorMetrics:
    """Outcome counts for a single confidence floor. Rates stay derived."""

    floor: float
    completed: int
    escalated: int
    labeled: int
    correct: int
    false_accepts: int
    false_rejects: int

    @property
    def escalation_rate(self) -> float | None:
        return self.escalated / self.completed if self.completed else None

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.labeled if self.labeled else None


def load_sweep_cases(payload: Any) -> list[SweepCase]:
    """Build replayable cases from an evaluation payload.

    Accepts the evaluator's current object output and the original bare list of
    reports. The list form carries no labels, so accuracy is unavailable for it.
    """
    if isinstance(payload, dict):
        records = [
            case
            for case in payload.get("cases", [])
            if case.get("status") == "completed"
            and isinstance(case.get("report"), dict)
            and "all_requirements" in case["report"]
        ]
    else:
        records = [{"report": report} for report in payload]

    cases: list[SweepCase] = []
    for index, record in enumerate(records):
        report = record["report"]
        cases.append(
            SweepCase(
                case_id=str(record.get("case_id", f"case_{index + 1}")),
                analyses=[
                    RequirementAnalysis(**analysis)
                    for analysis in report.get("all_requirements", [])
                ],
                recorded_verdict=str(report.get("verdict", "needs_review")),
                expected_verdict=record.get("expected_verdict"),
            )
        )
    return cases


def measure_floor(cases: list[SweepCase], floor: float) -> FloorMetrics:
    escalated = correct = labeled = false_accepts = false_rejects = 0
    for case in cases:
        verdict = case.verdict_at(floor)
        escalated += verdict == "needs_review"

        expected = case.expected_verdict
        if not expected:
            continue
        labeled += 1
        correct += verdict == expected
        false_accepts += verdict == "accept" and expected != "accept"
        false_rejects += verdict == "reject" and expected != "reject"

    return FloorMetrics(
        floor=floor,
        completed=len(cases),
        escalated=escalated,
        labeled=labeled,
        correct=correct,
        false_accepts=false_accepts,
        false_rejects=false_rejects,
    )


def _rate(numerator: int, denominator: int) -> str:
    if not denominator:
        return f"    n/a ({numerator}/{denominator})"
    return f"{numerator / denominator * 100:6.1f}% ({numerator}/{denominator})"


def format_report(metrics: list[FloorMetrics], independent: list[str]) -> str:
    header = (
        f"{'Floor':>6} | {'Verdict accuracy':>19} | {'False acc':>9} | "
        f"{'False rej':>9} | {'Escalation':>19}"
    )
    lines = [header, "-" * len(header)]
    for item in metrics:
        lines.append(
            f"{item.floor:>6.1f} | {_rate(item.correct, item.labeled):>19} | "
            f"{item.false_accepts:>9} | {item.false_rejects:>9} | "
            f"{_rate(item.escalated, item.completed):>19}"
        )

    if independent:
        lines.extend(
            [
                "",
                f"{len(independent)} case(s) scored no requirements and are "
                "threshold-independent;",
                "their recorded verdict is used at every floor rather than "
                "re-routed: " + ", ".join(independent) + ".",
            ]
        )
    return "\n".join(lines)


def sweep(reports_path: str, floors: list[float] | None = None) -> str:
    floors = floors or DEFAULT_FLOORS
    with open(reports_path, encoding="utf-8") as handle:
        payload = json.load(handle)

    cases = load_sweep_cases(payload)
    metrics = [measure_floor(cases, floor) for floor in floors]
    independent = [case.case_id for case in cases if case.threshold_independent]
    return format_report(metrics, independent)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", help="Path to an evaluation results JSON file")
    parser.add_argument(
        "--floors",
        type=float,
        nargs="+",
        help=f"Confidence floors to sweep (default: {DEFAULT_FLOORS})",
    )
    args = parser.parse_args(argv)
    print(sweep(args.reports, args.floors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
