import json

import pytest

from evaluation.run_evaluation import (
    calculate_metrics,
    load_cases,
    run_cases,
    validate_case,
    write_json,
    write_summary,
    parse_args,
)


def _case(resume_path: str = "resume.pdf") -> dict:
    return {
        "case_id": "case_001",
        "description": "A labeled case",
        "resume_path": resume_path,
        "job_description": "Python required",
        "expected_verdict": "accept",
        "requirements": [
            {
                "id": "R1",
                "requirement": "Python",
                "importance": "required",
                "ground_truth_score": 1.0,
            }
        ],
    }


def test_load_cases_reports_bad_json_line(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text('{"case_id": "ok"}\nnot-json\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"cases\.jsonl:2"):
        load_cases(path)


def test_validate_case_accepts_resume_relative_to_cases_file(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    (tmp_path / "resume.pdf").write_bytes(b"pdf")

    assert validate_case(_case(), cases_path) == []


def test_run_cases_skips_invalid_and_keeps_completed_case(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    (tmp_path / "resume.pdf").write_bytes(b"pdf")
    cases = [_case(), {**_case("missing.pdf"), "case_id": "case_002"}]

    def fake_runner(pdf_bytes, job_description):
        assert pdf_bytes == b"pdf"
        assert job_description == "Python required"
        return {"report": {"match_score": 100, "verdict": "accept"}}

    results = run_cases(cases, cases_path, "full", runner=fake_runner)

    assert [result["status"] for result in results] == ["completed", "skipped"]
    assert results[0]["report"]["verdict"] == "accept"
    assert "resume not found" in results[1]["issues"][0]


def test_calculate_full_metrics():
    results = [
        {
            "case_id": "case_001",
            "description": "match",
            "status": "completed",
            "expected_verdict": "accept",
            "requirements": _case()["requirements"],
            "report": {
                "match_score": 90,
                "verdict": "accept",
            },
            "diagnostics": {
                "evidence_quote_count": 2,
                "invalid_evidence_quote_count": 1,
                "positive_requirement_count": 1,
                "unsupported_positive_count": 0,
                "raised_correction_count": 1,
                "lowered_correction_count": 0,
            },
        },
        {
            "case_id": "case_002",
            "description": "missing",
            "status": "completed",
            "expected_verdict": "reject",
            "requirements": [],
            "report": {"match_score": 0, "verdict": "needs_review"},
            "diagnostics": {
                "evidence_quote_count": 0,
                "invalid_evidence_quote_count": 0,
                "positive_requirement_count": 0,
                "unsupported_positive_count": 0,
                "raised_correction_count": 0,
                "lowered_correction_count": 1,
            },
        },
    ]

    metrics = calculate_metrics(results, "full")

    assert metrics["verdict_accuracy"] == 0.5
    assert metrics["escalation_rate"] == 0.5
    assert metrics["score_mae"] == 10.0
    assert metrics["hallucinated_quote_rate"] == 0.5
    assert metrics["raised_correction_count"] == 1
    assert metrics["lowered_correction_count"] == 1


def test_writers_create_machine_and_human_readable_outputs(tmp_path):
    payload = {
        "mode": "full",
        "metrics": {"completed_cases": 1, "score_mae": None},
        "cases": [
            {
                "case_id": "case_001",
                "description": "A case",
                "status": "completed",
            }
        ],
    }
    json_path = tmp_path / "nested" / "results.json"
    summary_path = tmp_path / "nested" / "results.md"

    write_json(payload, json_path)
    write_summary(payload, summary_path)

    assert json.loads(json_path.read_text())["mode"] == "full"
    summary = summary_path.read_text()
    assert "# Evaluation Results — full mode" in summary
    assert "Score Mae | N/A" in summary


def test_case_id_option_can_be_repeated():
    args = parse_args(["--case-id", "case_001", "--case-id", "case_004"])

    assert args.case_id == ["case_001", "case_004"]
