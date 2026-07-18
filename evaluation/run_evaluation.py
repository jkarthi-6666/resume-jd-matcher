"""Evaluation harness. Supports --mode=naive for the Phase 1 baseline."""
import argparse
import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.config import validate_config


def run():
    parser = argparse.ArgumentParser(description="Run evaluation")
    parser.add_argument("--mode", choices=["naive", "full"], default="full")
    parser.add_argument("--cases", type=str, default="evaluation/labeled_set.jsonl")
    parser.add_argument("--output", type=str, default="evaluation/results.md")
    args = parser.parse_args()

    validate_config()

    cases = []
    with open(args.cases) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))

    print(f"Running {args.mode} evaluation on {len(cases)} cases…")

    metrics = {
        "hallucinated_quote_rate": [],
        "unsupported_match_rate": [],
        "escalation_rate": [],
        "cases": [],
    }

    for case in cases:
        print(f"  Case {case['case_id']}: {case['description']}")
        # Note: real eval requires actual PDF resumes.
        # This harness is structured for when sample PDFs are available.
        # For now, record the case structure.
        metrics["cases"].append({
            "case_id": case["case_id"],
            "description": case["description"],
            "mode": args.mode,
            "status": "needs_pdf",
        })

    _write_results(metrics, args.output, args.mode)
    print(f"Results written to {args.output}")


def _write_results(metrics: dict, output_path: str, mode: str) -> None:
    lines = [
        f"# Evaluation Results — {mode} mode",
        "",
        "## Cases",
        "",
    ]
    for case in metrics["cases"]:
        lines.append(f"- **{case['case_id']}**: {case['description']} — {case['status']}")

    lines += [
        "",
        "## Ablation Table",
        "",
        "| System | Unsupported match rate | Hallucinated quote rate | Cost / report |",
        "|---|---|---|---|",
        "| Phase 1 — single call, no RAG | ? | ? | ? |",
        "| + planning + hybrid RAG | ? | ? | ? |",
        "| + evidence validation | ? | ? | ? |",
        "| + reflection | ? | ? | ? |",
        "| + confidence routing | ? | ? | ? |",
        "",
        "_Fill with real numbers from the eval set._",
    ]

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    run()
