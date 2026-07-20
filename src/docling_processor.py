"""Docling-based PDF extraction and structure-aware resume chunking."""
from functools import lru_cache
from io import BytesIO
from typing import Any

from src.schemas import Chunk

__all__ = ["extract_and_chunk_resume"]

_MAX_PDF_BYTES = 25 * 1024 * 1024
_DOCUMENT_TIMEOUT_SECONDS = 120.0


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
        header_detected=bool(headings),
    )


def extract_and_chunk_resume(
    pdf_bytes: bytes,
    source: str = "resume.pdf",
) -> tuple[str, list[Chunk]]:
    """Convert an in-memory PDF and return evidence-safe text and app chunks.

    The returned full text is assembled from the same contextualized strings
    used for retrieval and scoring. This preserves the existing evidence
    contract: anything a scorer can quote from a chunk is also present when the
    reflector validates a correction against the full resume.
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
    except Exception as exc:
        raise ValueError(f"Could not process resume PDF with Docling: {exc}") from exc

    if not chunks:
        raise ValueError(
            "Docling could not extract usable text from the resume PDF."
        )

    resume_text = "\n\n".join(chunk.embed_text for chunk in chunks).strip()
    return resume_text, chunks
