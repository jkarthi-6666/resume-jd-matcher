"""Generate deterministic, privacy-safe PDF fixtures from labeled_set.jsonl."""
from __future__ import annotations

import argparse
import json
import pathlib
import textwrap

ROOT = pathlib.Path(__file__).resolve().parent.parent


def render_pdf(text: str, output_path: pathlib.Path) -> None:
    """Write a deterministic, text-only PDF using only the standard library."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered_lines: list[str] = []
    for raw_line in text.splitlines():
        lines = textwrap.wrap(raw_line, width=88) or [""]
        rendered_lines.extend(lines)
        if not raw_line:
            rendered_lines.append("")

    pages = [rendered_lines[i:i + 48] for i in range(0, len(rendered_lines), 48)] or [[]]
    font_id = 3 + 2 * len(pages)
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        (
            f"<< /Type /Pages /Count {len(pages)} /Kids "
            f"[{' '.join(f'{3 + 2 * i} 0 R' for i in range(len(pages)))}] >>"
        ).encode("ascii"),
    ]
    for index, lines in enumerate(pages):
        page_id = 3 + 2 * index
        content_id = page_id + 1
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
                f"/Contents {content_id} 0 R >>"
            ).encode("ascii")
        )
        commands = ["BT", "/F1 10 Tf", "54 738 Td", "14 TL"]
        for line in lines:
            safe = (
                line.encode("ascii", "replace").decode("ascii")
                .replace("\\", "\\\\")
                .replace("(", "\\(")
                .replace(")", "\\)")
            )
            commands.extend((f"({safe}) Tj", "T*"))
        commands.append("ET")
        stream = "\n".join(commands).encode("ascii")
        objects.append(
            f"<< /Length {len(stream)} >>\nstream\n".encode("ascii")
            + stream
            + b"\nendstream"
        )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    payload = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for object_id, body in enumerate(objects, start=1):
        offsets.append(len(payload))
        payload.extend(f"{object_id} 0 obj\n".encode("ascii"))
        payload.extend(body)
        payload.extend(b"\nendobj\n")
    xref_offset = len(payload)
    payload.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    payload.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        payload.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    payload.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    output_path.write_bytes(payload)


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
