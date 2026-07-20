from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.docling_processor import _to_app_chunk, extract_and_chunk_resume
from src.validate import normalize


class FakeDoclingChunk:
    def __init__(self, text: str, headings: list[str] | None = None):
        self.text = text
        self.meta = SimpleNamespace(headings=headings or [])


class FakeChunker:
    def __init__(self, chunks):
        self.chunks = chunks
        self.document_seen = None

    def chunk(self, dl_doc):
        self.document_seen = dl_doc
        return iter(self.chunks)

    def contextualize(self, chunk):
        headings = "\n".join(chunk.meta.headings)
        return "\n".join(part for part in (headings, chunk.text) if part)


class FakeConverter:
    def __init__(self, document=None, error: Exception | None = None):
        self.document = document or object()
        self.error = error
        self.source_seen = None
        self.max_file_size_seen = None

    def convert(self, source, max_file_size):
        if self.error:
            raise self.error
        self.source_seen = source
        self.max_file_size_seen = max_file_size
        return SimpleNamespace(document=self.document)


def test_mapping_accepts_real_docling_hybrid_chunk():
    """Guard the adapter against changes in Docling's real chunk metadata API."""
    from docling.chunking import HybridChunker
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter

    document = DocumentConverter().convert_string(
        "# Experience\n\n## Senior Engineer\n\nBuilt Python APIs.",
        format=InputFormat.MD,
        name="resume",
    ).document
    chunker = HybridChunker()
    docling_chunk = next(chunker.chunk(dl_doc=document))

    chunk = _to_app_chunk(docling_chunk, chunker, 0, "resume.pdf")

    assert chunk.section == "Experience"
    assert chunk.header == "Senior Engineer"
    assert chunk.body == "Built Python APIs."
    assert chunk.embed_text == "Experience\nSenior Engineer\nBuilt Python APIs."


def test_extract_and_chunk_resume_maps_docling_chunks_to_app_contract():
    converter = FakeConverter()
    chunker = FakeChunker([
        FakeDoclingChunk(
            "Built Python APIs for production workloads.",
            ["Experience", "Senior Engineer"],
        ),
        FakeDoclingChunk("Python, FastAPI, PostgreSQL", ["Skills"]),
    ])

    with patch("src.docling_processor._get_converter", return_value=converter), \
         patch("src.docling_processor._get_chunker", return_value=chunker):
        resume_text, chunks = extract_and_chunk_resume(
            b"not-read-by-the-fake-converter",
            source="candidate.pdf",
        )

    assert [chunk.chunk_id for chunk in chunks] == ["chunk_000", "chunk_001"]
    assert chunks[0].section == "Experience"
    assert chunks[0].header == "Senior Engineer"
    assert chunks[0].header_detected is True
    assert chunks[0].body == "Built Python APIs for production workloads."
    assert chunks[0].embed_text == (
        "Experience\nSenior Engineer\nBuilt Python APIs for production workloads."
    )
    assert all(chunk.source == "candidate.pdf" for chunk in chunks)
    assert chunker.document_seen is converter.document
    assert converter.source_seen.name == "candidate.pdf"

    # Reflection receives the same serialization that scoring can quote.
    for chunk in chunks:
        assert normalize(chunk.embed_text) in normalize(resume_text)


def test_extract_and_chunk_resume_supplies_fallback_metadata_and_skips_empty_chunks():
    converter = FakeConverter()
    chunker = FakeChunker([
        FakeDoclingChunk("   ", ["Empty"]),
        FakeDoclingChunk("Unstructured resume content"),
    ])

    with patch("src.docling_processor._get_converter", return_value=converter), \
         patch("src.docling_processor._get_chunker", return_value=chunker):
        _, chunks = extract_and_chunk_resume(b"pdf", source="upload")

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "chunk_000"
    assert chunks[0].section == "Unknown"
    assert chunks[0].header == "Chunk 1"
    assert chunks[0].header_detected is False
    assert chunks[0].source == "upload"
    assert converter.source_seen.name == "upload.pdf"


def test_extract_and_chunk_resume_rejects_empty_input_before_conversion():
    with patch("src.docling_processor._get_converter") as get_converter:
        with pytest.raises(ValueError, match="Resume PDF is empty"):
            extract_and_chunk_resume(b"")
    get_converter.assert_not_called()


def test_extract_and_chunk_resume_rejects_a_document_without_text_chunks():
    with patch("src.docling_processor._get_converter", return_value=FakeConverter()), \
         patch("src.docling_processor._get_chunker", return_value=FakeChunker([])):
        with pytest.raises(ValueError, match="could not extract usable text"):
            extract_and_chunk_resume(b"pdf")


def test_extract_and_chunk_resume_wraps_docling_conversion_failures():
    converter = FakeConverter(error=RuntimeError("conversion failed"))

    with patch("src.docling_processor._get_converter", return_value=converter):
        with pytest.raises(
            ValueError,
            match="Could not process resume PDF with Docling: conversion failed",
        ):
            extract_and_chunk_resume(b"pdf")
