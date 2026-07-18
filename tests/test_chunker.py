"""Unit tests for resume chunking."""
import pytest
from src.chunker import chunk_resume


SAMPLE_RESUME = """
Summary
Experienced software engineer with 5+ years in Python and distributed systems.

Experience
Senior Engineer, Acme Corp, Jan 2021 - Present
Led development of microservices platform serving 10M requests/day.
Built CI/CD pipelines using GitHub Actions and Docker.

Junior Engineer, StartupXYZ, Jun 2018 - Dec 2020
Built REST APIs with Flask and FastAPI.
Managed PostgreSQL databases.

Projects
Kubernetes Migration, 2022 - 2023
Orchestrated migration of 20 services to Kubernetes on GCP.

Education
B.S. Computer Science, MIT, 2014 - 2018
GPA 3.9. Coursework in algorithms and systems.

Skills
Python, Go, Docker, Kubernetes, PostgreSQL, Redis, AWS, GCP
"""


class TestChunkResume:
    def test_returns_chunks(self):
        chunks = chunk_resume(SAMPLE_RESUME)
        assert len(chunks) > 0

    def test_no_empty_bodies(self):
        chunks = chunk_resume(SAMPLE_RESUME)
        for c in chunks:
            assert c.body.strip(), f"Empty body in chunk {c.chunk_id}"

    def test_unique_chunk_ids(self):
        chunks = chunk_resume(SAMPLE_RESUME)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_embed_text_contains_body(self):
        chunks = chunk_resume(SAMPLE_RESUME)
        for c in chunks:
            assert c.body.strip()[:20] in c.embed_text or c.header in c.embed_text

    def test_skills_section_present(self):
        chunks = chunk_resume(SAMPLE_RESUME)
        sections = [c.section for c in chunks]
        assert any("Skills" in s for s in sections)

    def test_fallback_for_unstructured_text(self):
        unstructured = "This is a resume with no clear sections.\n\nJust some text about Python and Docker.\n\nAnd more text here."
        chunks = chunk_resume(unstructured)
        assert len(chunks) > 0
