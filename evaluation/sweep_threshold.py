"""Sweep confidence threshold and plot escalation-precision curve."""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import json
import argparse
from src.router import route


def sweep(reports_path: str, floors: list[float] | None = None) -> None:
    if floors is None:
        floors = [round(x * 0.1, 1) for x in range(5, 10)]  # 0.5 to 0.9

    with open(reports_path) as f:
        reports = json.load(f)

    print(f"{'Floor':>8} | {'Escalation%':>12} | {'Notes'}")
    print("-" * 50)
    for floor in floors:
        escalated = 0
        for report_data in reports:
            from src.schemas import RequirementAnalysis
            analyses = [RequirementAnalysis(**r) for r in report_data["all_requirements"]]
            verdict, _ = route(analyses, confidence_floor=floor)
            if verdict == "needs_review":
                escalated += 1
        rate = escalated / len(reports) * 100 if reports else 0
        print(f"{floor:>8.1f} | {rate:>11.1f}% | {escalated}/{len(reports)} escalated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", help="Path to JSON file with list of FinalReport dicts")
    args = parser.parse_args()
    sweep(args.reports)
