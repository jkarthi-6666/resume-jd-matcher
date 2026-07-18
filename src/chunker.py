"""Entry-based resume chunking. One chunk per job, project, degree, or skills section."""
import re
from src.schemas import Chunk

# Section header patterns
_SECTION_PATTERNS = [
    (r"(?i)^(experience|work experience|employment|professional experience)", "Experience"),
    (r"(?i)^(projects?|personal projects?|side projects?)",                  "Projects"),
    (r"(?i)^(education|academic background|qualifications)",                 "Education"),
    (r"(?i)^(skills?|technical skills?|core competencies|technologies)",     "Skills"),
    (r"(?i)^(summary|objective|profile|about me)",                           "Summary"),
    (r"(?i)^(certifications?|awards?|publications?|achievements?)",          "Certifications"),
]

# Date range patterns (e.g. "Jan 2020 – Dec 2022", "2019 - Present")
_DATE_RE = re.compile(
    r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|"
    r"April|June|July|August|September|October|November|December)?\s*\d{4}"
    r"\s*[-–]\s*(Present|Current|\d{4}|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|"
    r"Nov|Dec|January|February|March|April|June|July|August|September|October|"
    r"November|December)\b",
    re.IGNORECASE,
)


def chunk_resume(text: str, source: str = "resume.pdf") -> list[Chunk]:
    lines = text.splitlines()
    sections = _split_into_sections(lines)
    chunks: list[Chunk] = []
    chunk_idx = 0

    for section_name, section_lines in sections:
        if section_name == "Skills" or section_name == "Summary" or section_name == "Certifications":
            # Treat whole section as one chunk
            body = "\n".join(section_lines).strip()
            if not body:
                continue
            header = section_name
            chunk = Chunk(
                chunk_id=f"chunk_{chunk_idx:03d}",
                section=section_name,
                header=header,
                body=body,
                embed_text=f"{header}\n{body}",
                source=source,
                header_detected=True,
            )
            chunks.append(chunk)
            chunk_idx += 1
        else:
            # Split into individual entries
            entries = _split_into_entries(section_lines, section_name)
            for entry_header, entry_body in entries:
                if not entry_body.strip():
                    continue
                chunk = Chunk(
                    chunk_id=f"chunk_{chunk_idx:03d}",
                    section=section_name,
                    header=entry_header,
                    body=entry_body,
                    embed_text=f"{entry_header}\n{entry_body}",
                    source=source,
                    header_detected=True,
                )
                chunks.append(chunk)
                chunk_idx += 1

    if not chunks:
        # Fallback: paragraph-based chunks
        chunks = _fallback_chunks(text, source)

    return [c for c in chunks if c.body.strip()]


def _split_into_sections(lines: list[str]) -> list[tuple[str, list[str]]]:
    """Group lines into (section_name, lines) tuples."""
    sections: list[tuple[str, list[str]]] = []
    current_section = "Unknown"
    current_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        matched_section = _match_section_header(stripped)
        if matched_section:
            if current_lines:
                sections.append((current_section, current_lines))
            current_section = matched_section
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_section, current_lines))

    return sections


def _match_section_header(line: str) -> str | None:
    for pattern, name in _SECTION_PATTERNS:
        if re.match(pattern, line.strip()):
            return name
    return None


def _split_into_entries(lines: list[str], section: str) -> list[tuple[str, str]]:
    """Split section lines into individual job/project/education entries."""
    entries: list[tuple[str, str]] = []
    current_header = section
    current_body_lines: list[str] = []
    found_any_entry = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            current_body_lines.append(line)
            continue

        # An entry boundary: a line that contains a date range
        if _DATE_RE.search(stripped) and len(stripped) < 200:
            if found_any_entry and current_body_lines:
                body = "\n".join(current_body_lines).strip()
                entries.append((current_header, body))
                current_body_lines = []
            current_header = stripped
            found_any_entry = True
        else:
            current_body_lines.append(line)

    if current_body_lines:
        body = "\n".join(current_body_lines).strip()
        if body:
            entries.append((current_header, body))

    if not entries:
        # No date boundaries found — treat whole section as one entry
        body = "\n".join(lines).strip()
        entries = [(section, body)]

    return entries


def _fallback_chunks(text: str, source: str) -> list[Chunk]:
    """Paragraph-based fallback when section detection fails."""
    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    for i, para in enumerate(paragraphs):
        para = para.strip()
        if len(para) < 30:
            continue
        chunks.append(Chunk(
            chunk_id=f"chunk_{i:03d}",
            section="Unknown",
            header=f"Paragraph {i+1}",
            body=para,
            embed_text=para,
            source=source,
            header_detected=False,
        ))
    return chunks
