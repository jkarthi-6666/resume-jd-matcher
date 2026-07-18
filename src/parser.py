"""Extract text from resume PDFs."""
import re
import fitz  # PyMuPDF


class ImageOnlyPDF(Exception):
    pass


def extract_text(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    text = "\n".join(pages)
    text = _clean(text)
    if len(text.strip()) < 200:
        raise ImageOnlyPDF(
            "Extracted text is under 200 characters. "
            "This PDF may be image-only. OCR is not supported in v1."
        )
    return text


def _clean(text: str) -> str:
    # Fix common PDF ligature artifacts
    replacements = {
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "–": "-",
        "—": "-",
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
    }
    for orig, rep in replacements.items():
        text = text.replace(orig, rep)
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
