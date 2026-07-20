"""Run labeled resume/JD evaluations and emit JSON plus a Markdown summary."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from datetime import datetime, timezone
from typing import Any, Callable

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import validate_config


CaseRunner = Callable[[bytes, str], dict[str, Any]]


def load_cases(path: pathlib.Path) -> list[dict[str, Any]]:
    """Load JSONL cases and add a useful line number to parse errors."""
    cases: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
            if not isinstance(case, dict):
                raise ValueError(f"Case on {path}:{line_number} must be a JSON object.")
            cases.append(case)
    return cases


def resolve_resume_path(value: str, cases_path: pathlib.Path) -> pathlib.Path:
    """Resolve paths relative to either the JSONL file or repository root."""
    path = pathlib.Path(value).expanduser()
    if path.is_absolute():
        return path
    relative_to_cases = cases_path.parent / path
    if relative_to_cases.exists():
        return relative_to_cases
    return ROOT / path


def validate_case(case: dict[str, Any], cases_path: pathlib.Path) -> list[str]:
    issues: list[str] = []
    for field in ("case_id", "description", "job_description", "resume_path"):
        if not str(case.get(field, "")).strip():
            issues.append(f"missing {field}")

    resume_path = str(case.get("resume_path", "")).strip()
    if resume_path and not resolve_resume_path(resume_path, cases_path).is_file():
        issues.append(f"resume not found: {resume_path}")

    expected = case.get("expected_verdict")
    if expected is not None and expected not in {"accept", "needs_review", "reject"}:
        issues.append(f"invalid expected_verdict: {expected!r}")
    return issues


def _full_runner(pdf_bytes: bytes, job_description: str) -> dict[str, Any]:
    from src.pipeline import run_full
    from src.schemas import Chunk
    from src.validate import invalid_evidence_quotes

    report, debug = run_full(pdf_bytes, job_description)
    resume_chunk = Chunk(
        chunk_id="evaluation_resume",
        section="Full",
        header="Full resume",
        body=debug.validation_corpus,
        embed_text=debug.validation_corpus,
        source="evaluation",
    )
    quotes = [quote for analysis in report.all_requirements for quote in analysis.evidence]
    invalid_quotes = invalid_evidence_quotes(quotes, [resume_chunk])
    unsupported_positive = sum(
        1
        for analysis in report.all_requirements
        if analysis.score > 0 and analysis.evidence_valid is not True
    )
    positive_requirements = sum(a.score > 0 for a in report.all_requirements)

    return {
        "report": report.model_dump(),
        "diagnostics": {
            "evidence_quote_count": len(quotes),
            "invalid_evidence_quote_count": len(invalid_quotes),
            "positive_requirement_count": positive_requirements,
            "unsupported_positive_count": unsupported_positive,
            "raised_correction_count": sum(
                correction.direction == "raised" for correction in report.corrections
            ),
            "lowered_correction_count": sum(
                correction.direction == "lowered" for correction in report.corrections
            ),
            "rejected_correction_reasons": [
                rejection.reason
                for rejection in (
                    debug.reflection_result.rejected_corrections
                    if debug.reflection_result else []
                )
            ],
            "planner_dropped_requirement_count": (
                debug.requirement_plan.dropped_requirement_count
                if debug.requirement_plan else 0
            ),
            "planner_output_requirement_count": (
                len(debug.requirement_plan.requirements)
                + debug.requirement_plan.dropped_requirement_count
                if debug.requirement_plan else 0
            ),
        },
    }


def _naive_runner(pdf_bytes: bytes, job_description: str) -> dict[str, Any]:
    from src.pipeline import run_naive

    return {"report": run_naive(pdf_bytes, job_description).model_dump()}


def run_cases(
    cases: list[dict[str, Any]],
    cases_path: pathlib.Path,
    mode: str,
    runner: CaseRunner | None = None,
) -> list[dict[str, Any]]:
    """Run valid cases independently so one provider failure does not lose a run."""
    runner = runner or (_full_runner if mode == "full" else _naive_runner)
    results: list[dict[str, Any]] = []

    for case in cases:
        case_id = str(case.get("case_id", "unknown"))
        issues = validate_case(case, cases_path)
        base = {
            "case_id": case_id,
            "description": case.get("description", ""),
            "expected_verdict": case.get("expected_verdict"),
            "requirements": case.get("requirements", []),
        }
        if issues:
            results.append({**base, "status": "skipped", "issues": issues})
            print(f"  SKIP {case_id}: {'; '.join(issues)}")
            continue

        print(f"  RUN  {case_id}: {case['description']}")
        try:
            pdf_path = resolve_resume_path(case["resume_path"], cases_path)
            payload = runner(pdf_path.read_bytes(), case["job_description"])
            results.append({**base, "status": "completed", **payload})
        except Exception as exc:  # preserve the rest of a potentially costly run
            results.append(
                {
                    **base,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            print(f"  FAIL {case_id}: {type(exc).__name__}: {exc}")
    return results


def _expected_score(case_result: dict[str, Any]) -> float | None:
    requirements = case_result.get("requirements", [])
    labeled = [
        r for r in requirements
        if "ground_truth_score" in r and r.get("kind", "scored") != "gate"
    ]
    if not labeled:
        return None
    weights = {"required": 2, "preferred": 1, "unknown": 1}
    possible = sum(weights.get(r.get("importance", "unknown"), 1) for r in labeled)
    earned = sum(
        float(r["ground_truth_score"])
        * weights.get(r.get("importance", "unknown"), 1)
        for r in labeled
    )
    return round(earned / possible * 100, 1) if possible else None


def calculate_metrics(results: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    completed = [item for item in results if item["status"] == "completed"]
    metrics: dict[str, Any] = {
        "total_cases": len(results),
        "completed_cases": len(completed),
        "skipped_cases": sum(item["status"] == "skipped" for item in results),
        "failed_cases": sum(item["status"] == "failed" for item in results),
    }

    score_errors: list[float] = []
    for item in completed:
        expected_score = _expected_score(item)
        if expected_score is not None:
            score_errors.append(abs(float(item["report"]["match_score"]) - expected_score))
    metrics["score_mae"] = round(sum(score_errors) / len(score_errors), 2) if score_errors else None

    if mode == "naive":
        return metrics

    labeled = [item for item in completed if item.get("expected_verdict")]
    correct = sum(
        item["report"]["verdict"] == item["expected_verdict"] for item in labeled
    )
    metrics.update(
        {
            "verdict_accuracy": round(correct / len(labeled), 4) if labeled else None,
            "escalation_rate": round(
                sum(item["report"]["verdict"] == "needs_review" for item in completed)
                / len(completed),
                4,
            )
            if completed
            else None,
            "false_accept_count": sum(
                item["report"]["verdict"] == "accept"
                and item["expected_verdict"] != "accept"
                for item in labeled
            ),
            "false_reject_count": sum(
                item["report"]["verdict"] == "reject"
                and item["expected_verdict"] != "reject"
                for item in labeled
            ),
        }
    )

    quote_count = sum(item.get("diagnostics", {}).get("evidence_quote_count", 0) for item in completed)
    invalid_count = sum(
        item.get("diagnostics", {}).get("invalid_evidence_quote_count", 0)
        for item in completed
    )
    positive_count = sum(
        item.get("diagnostics", {}).get("positive_requirement_count", 0)
        for item in completed
    )
    unsupported_count = sum(
        item.get("diagnostics", {}).get("unsupported_positive_count", 0)
        for item in completed
    )
    metrics.update(
        {
            "hallucinated_quote_rate": round(invalid_count / quote_count, 4) if quote_count else 0.0,
            "unsupported_match_rate": round(unsupported_count / positive_count, 4)
            if positive_count
            else 0.0,
            "raised_correction_count": sum(
                item.get("diagnostics", {}).get("raised_correction_count", 0)
                for item in completed
            ),
            "lowered_correction_count": sum(
                item.get("diagnostics", {}).get("lowered_correction_count", 0)
                for item in completed
            ),
        }
    )
    planner_outputs = sum(
        item.get("diagnostics", {}).get("planner_output_requirement_count", 0)
        for item in completed
    )
    planner_dropped = sum(
        item.get("diagnostics", {}).get("planner_dropped_requirement_count", 0)
        for item in completed
    )
    rejection_reason_counts: dict[str, int] = {}
    for item in completed:
        for reason in item.get("diagnostics", {}).get(
            "rejected_correction_reasons", []
        ):
            rejection_reason_counts[reason] = rejection_reason_counts.get(reason, 0) + 1
    metrics["planner_hallucination_rate"] = (
        round(planner_dropped / planner_outputs, 4) if planner_outputs else 0.0
    )
    metrics["rejected_correction_reason_counts"] = rejection_reason_counts
    return metrics


def write_json(payload: dict[str, Any], output_path: pathlib.Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_summary(payload: dict[str, Any], summary_path: pathlib.Path) -> None:
    metrics = payload["metrics"]
    lines = [
        f"# Evaluation Results — {payload['mode']} mode",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for name, value in metrics.items():
        rendered = "N/A" if value is None else str(value)
        lines.append(f"| {name.replace('_', ' ').title()} | {rendered} |")

    lines.extend(["", "## Cases", ""])
    for item in payload["cases"]:
        detail = item.get("error") or "; ".join(item.get("issues", []))
        suffix = f" — {detail}" if detail else ""
        lines.append(
            f"- **{item['case_id']}**: {item['description']} — {item['status']}{suffix}"
        )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run resume matcher evaluation")
    parser.add_argument("--mode", choices=["naive", "full"], default="full")
    parser.add_argument("--cases", default="evaluation/labeled_set.jsonl")
    parser.add_argument("--output", default="evaluation/results.json")
    parser.add_argument("--summary", default="evaluation/results.md")
    parser.add_argument(
        "--case-id",
        action="append",
        help="Run one case ID. Repeat the option to select multiple cases.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Check dataset paths and labels without calling a model provider.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cases_path = pathlib.Path(args.cases).resolve()
    cases = load_cases(cases_path)
    if args.case_id:
        requested = set(args.case_id)
        cases = [case for case in cases if case.get("case_id") in requested]
        found = {case.get("case_id") for case in cases}
        missing = sorted(requested - found)
        if missing:
            print(f"Unknown case ID(s): {', '.join(missing)}")
            return 2
    print(f"Loaded {len(cases)} case(s) from {cases_path}")

    if args.validate_only:
        invalid = 0
        for case in cases:
            issues = validate_case(case, cases_path)
            if issues:
                invalid += 1
                print(f"  INVALID {case.get('case_id', 'unknown')}: {'; '.join(issues)}")
            else:
                print(f"  READY   {case['case_id']}")
        print(f"Validation complete: {len(cases) - invalid} ready, {invalid} invalid.")
        return 1 if invalid else 0

    validate_config()
    print(f"Running {args.mode} evaluation…")
    results = run_cases(cases, cases_path, args.mode)
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "metrics": calculate_metrics(results, args.mode),
        "cases": results,
    }
    output_path = pathlib.Path(args.output)
    summary_path = pathlib.Path(args.summary)
    write_json(payload, output_path)
    write_summary(payload, summary_path)
    print(f"JSON results written to {output_path}")
    print(f"Summary written to {summary_path}")
    return 1 if payload["metrics"]["failed_cases"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
