"""Docling-based PDF extraction and structure-aware resume chunking."""
from functools import lru_cache
from io import BytesIO
from typing import Any

from src.schemas import Chunk
from src import config

__all__ = [
    "DocumentProcessingTimeoutError",
    "UnassessableDocumentError",
    "extract_and_chunk_resume",
]

_MAX_PDF_BYTES = 25 * 1024 * 1024
_DOCUMENT_TIMEOUT_SECONDS = 120.0


class UnassessableDocumentError(ValueError):
    """The document was processed, but does not contain assessable resume text."""


class DocumentProcessingTimeoutError(TimeoutError):
    """Docling exceeded its configured document-processing deadline."""


def _content_is_sufficient(text: str) -> bool:
    """Reject negligible or symbol-heavy OCR output without penalizing brevity."""
    non_whitespace = [char for char in text if not char.isspace()]
    if not non_whitespace:
        return False
    alphabetic = sum(char.isalpha() for char in non_whitespace)
    return (
        alphabetic >= config.MIN_RESUME_ALPHABETIC_CHARS
        and alphabetic / len(non_whitespace) >= config.MIN_RESUME_ALPHA_RATIO
    )


def _assemble_resume_text(document: Any, chunks: list[Chunk]) -> str:
    """Return a validation union of reading-order and contextualized text."""
    exporter = getattr(document, "export_to_text", None)
    contiguous = str(exporter()).strip() if callable(exporter) else ""
    if not contiguous:
        contiguous = "\n\n".join(chunk.body for chunk in chunks).strip()
    contextualized = "\n\n".join(chunk.embed_text for chunk in chunks).strip()
    return "\n\n".join(part for part in (contiguous, contextualized) if part)


@lru_cache(maxsize=1)
def _get_converter():
    """Initialize Docling's model-backed converter once per process."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions(
        do_ocr=True,
        document_timeout=_DOCUMENT_TIMEOUT_SECONDS,
    )
    return DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=options),
        },
    )


@lru_cache(maxsize=1)
def _get_chunker():
    """Initialize the structure- and token-aware chunker once per process."""
    from docling.chunking import HybridChunker

    return HybridChunker(merge_peers=True)


def _to_app_chunk(
    docling_chunk: Any,
    chunker: Any,
    index: int,
    source: str,
) -> Chunk:
    """Map one Docling chunk into the application's existing Chunk contract."""
    meta = getattr(docling_chunk, "meta", None)
    headings = [
        str(heading).strip()
        for heading in (getattr(meta, "headings", None) or [])
        if str(heading).strip()
    ]
    body = str(getattr(docling_chunk, "text", "")).strip()
    contextualized = str(chunker.contextualize(chunk=docling_chunk)).strip()

    return Chunk(
        chunk_id=f"chunk_{index:03d}",
        section=headings[0] if headings else "Unknown",
        header=headings[-1] if headings else f"Chunk {index + 1}",
        body=body,
        embed_text=contextualized or body,
        source=source,
    )


def extract_and_chunk_resume(
    pdf_bytes: bytes,
    source: str = "resume.pdf",
) -> tuple[str, list[Chunk]]:
    """Convert an in-memory PDF and return evidence-safe text and app chunks.

    The returned full text combines Docling's contiguous reading-order export
    with the contextualized strings used for retrieval and scoring. This both
    preserves cross-chunk prose and guarantees that anything a scorer can quote
    is present when the reflector validates a correction.
    """
    if not pdf_bytes:
        raise ValueError("Resume PDF is empty.")
    if len(pdf_bytes) > _MAX_PDF_BYTES:
        raise ValueError("Resume PDF exceeds the 25 MB processing limit.")

    try:
        from docling.datamodel.base_models import DocumentStream

        stream_name = source if source.lower().endswith(".pdf") else f"{source}.pdf"
        stream = DocumentStream(name=stream_name, stream=BytesIO(pdf_bytes))
        result = _get_converter().convert(
            stream,
            max_file_size=_MAX_PDF_BYTES,
        )
        if result.has_timeout_errors():
            raise DocumentProcessingTimeoutError(
                "Resume processing took too long. Try a smaller or simpler PDF."
            )
        document = result.document
        chunker = _get_chunker()

        chunks: list[Chunk] = []
        for docling_chunk in chunker.chunk(dl_doc=document):
            if not str(getattr(docling_chunk, "text", "")).strip():
                continue
            chunks.append(
                _to_app_chunk(
                    docling_chunk=docling_chunk,
                    chunker=chunker,
                    index=len(chunks),
                    source=source,
                )
            )
    except DocumentProcessingTimeoutError:
        raise
    except TimeoutError as exc:
        raise DocumentProcessingTimeoutError(
            "Resume processing took too long. Try a smaller or simpler PDF."
        ) from exc
    except Exception as exc:
        raise ValueError(f"Could not process resume PDF with Docling: {exc}") from exc

    if not chunks:
        raise UnassessableDocumentError(
            "We could not read usable text from this resume. Human review or a "
            "clearer, text-based PDF is required."
        )

    resume_text = _assemble_resume_text(document, chunks)
    if not _content_is_sufficient(resume_text):
        raise UnassessableDocumentError(
            "We could not read enough reliable text from this resume to assess "
            "qualifications. The PDF may be a poor-quality scan or contain "
            "non-linguistic OCR output; human review is required."
        )
    return resume_text, chunks
