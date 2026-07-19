"""Generate deterministic, privacy-safe PDF fixtures from labeled_set.jsonl."""
from __future__ import annotations

import argparse
import json
import pathlib
import textwrap

import fitz

ROOT = pathlib.Path(__file__).resolve().parent.parent


def render_pdf(text: str, output_path: pathlib.Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    page = document.new_page(width=612, height=792)
    y = 54.0

    for raw_line in text.splitlines():
        lines = textwrap.wrap(raw_line, width=88) or [""]
        for line in lines:
            if y > 738:
                page = document.new_page(width=612, height=792)
                y = 54.0
            page.insert_text((54, y), line, fontsize=10, fontname="helv")
            y += 14
        if not raw_line:
            y += 4

    metadata = document.metadata
    metadata.update(
        {
            "title": output_path.stem,
            "author": "Synthetic evaluation fixture",
            "subject": "Resume matcher evaluation",
        }
    )
    document.set_metadata(metadata)
    document.save(output_path, garbage=4, deflate=True)
    document.close()


def generate(cases_path: pathlib.Path) -> int:
    generated = 0
    with cases_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            case = json.loads(line)
            if not case.get("resume_text") or not case.get("resume_path"):
                raise ValueError(
                    f"Case on line {line_number} needs resume_text and resume_path."
                )
            output_path = pathlib.Path(case["resume_path"])
            if not output_path.is_absolute():
                output_path = ROOT / output_path
            render_pdf(case["resume_text"], output_path)
            print(f"Generated {output_path.relative_to(ROOT)}")
            generated += 1
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default="evaluation/labeled_set.jsonl")
    args = parser.parse_args()
    cases_path = pathlib.Path(args.cases)
    if not cases_path.is_absolute():
        cases_path = ROOT / cases_path
    count = generate(cases_path)
    print(f"Generated {count} synthetic PDF(s).")


if __name__ == "__main__":
    main()
