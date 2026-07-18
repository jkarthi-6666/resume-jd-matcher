"""Integration-level tests for the pipeline (all model calls mocked).

Mocks target src.llm.call, not a provider function. _call_raw dispatches on
LLM_PROVIDER, so patching src.llm._openai only works when LLM_PROVIDER happens
to be "openai" — under any other provider the patch is bypassed and the call
hits the network. See tests/conftest.py.
"""
import re

import pytest
from unittest.mock import patch, MagicMock

from src.schemas import (
    RequirementPlan, Requirement, RequirementAnalysis,
    RerankerResult, RerankedChunk, ReflectionResult, NaiveResult,
)
from tests.conftest import fake_embeddings


MOCK_RESUME = " ".join(["word"] * 50) + """
Experience
Senior Engineer, Acme, Jan 2020 - Present
Built Python REST APIs. Led Kubernetes migration.

Skills
Python, REST, Kubernetes, Docker
"""

MOCK_JD = "We need a Python developer with REST API experience."


def _fake_llm_call(prompt, model, schema, temperature=0.0, system=""):
    """Dispatch on the requested schema rather than on prompt wording.

    The previous version matched substrings of the prompt and fell through to a
    default when the prompt text drifted, so a stage could be silently mocked
    with the wrong payload.
    """
    if schema is RequirementPlan:
        return RequirementPlan(requirements=[
            Requirement(id="R1", requirement="Python experience",
                        category="technical_skill", importance="required"),
            Requirement(id="R2", requirement="REST API experience",
                        category="technical_skill", importance="required"),
        ])

    if schema is RerankerResult:
        ids = re.findall(r"chunk_id=(chunk_\d+)", prompt)
        return RerankerResult(ranked_chunks=[
            RerankedChunk(chunk_id=cid, justification="test") for cid in ids
        ])

    if schema is RequirementAnalysis:
        # scorer overwrites requirement_id/requirement/importance after the
        # call, so the identifying fields here are placeholders.
        return RequirementAnalysis(
            requirement_id="R1",
            requirement="test",
            importance="required",
            score=1.0,
            confidence=0.9,
            evidence=["Built Python REST APIs."],
            reason="Directly stated.",
            retrieved_chunk_ids=["chunk_000"],
        )

    if schema is ReflectionResult:
        return ReflectionResult(
            approved=True,
            changed_requirements=[],
            review_notes=["No issues found."],
        )

    if schema is NaiveResult:
        return NaiveResult(
            match_score=80,
            matched_skills=["Python", "REST"],
            missing_skills=[],
            recommendation="Proceed to interview.",
        )

    raise AssertionError(f"Unexpected schema requested: {schema}")


@pytest.fixture
def mock_pdf():
    page = MagicMock()
    page.get_text.return_value = MOCK_RESUME * 3
    doc = MagicMock()
    doc.__iter__ = MagicMock(return_value=iter([page]))
    doc.close = MagicMock()
    return doc


class TestPipelineMocked:
    @patch("src.embeddings.get_embeddings", side_effect=fake_embeddings)
    @patch("src.llm.call", side_effect=_fake_llm_call)
    @patch("fitz.open")
    def test_full_pipeline_runs(self, mock_fitz, mock_call, mock_embed, mock_pdf):
        mock_fitz.return_value = mock_pdf

        from src.pipeline import run_full
        from src.planner import clear_cache
        clear_cache()

        report, debug = run_full(b"fake_pdf", MOCK_JD)

        assert report.verdict in ("accept", "needs_review", "reject")
        assert 0 <= report.match_score <= 100
        assert report.score_pre_reflection >= 0
        assert len(report.all_requirements) == 2
        assert len(debug.chunks) > 0
        assert len(debug.requirements) == 2

    @patch("src.embeddings.get_embeddings", side_effect=fake_embeddings)
    @patch("src.llm.call", side_effect=_fake_llm_call)
    @patch("fitz.open")
    def test_full_pipeline_preserves_requirement_order(
        self, mock_fitz, mock_call, mock_embed, mock_pdf
    ):
        """Requirements are scored concurrently but must report in plan order."""
        mock_fitz.return_value = mock_pdf

        from src.pipeline import run_full
        from src.planner import clear_cache
        clear_cache()

        report, _ = run_full(b"fake_pdf", MOCK_JD)

        assert [a.requirement_id for a in report.all_requirements] == ["R1", "R2"]

    @patch("src.llm.call", side_effect=_fake_llm_call)
    @patch("fitz.open")
    def test_naive_mode_runs(self, mock_fitz, mock_call, mock_pdf):
        mock_fitz.return_value = mock_pdf

        from src.pipeline import run_naive

        result = run_naive(b"fake_pdf", MOCK_JD)

        assert 0 <= result.match_score <= 100
        assert "Python" in result.matched_skills

    @patch("src.llm.call", side_effect=_fake_llm_call)
    @patch("fitz.open")
    def test_empty_job_description_rejected(self, mock_fitz, mock_call, mock_pdf):
        mock_fitz.return_value = mock_pdf

        from src.pipeline import run_full

        with pytest.raises(ValueError, match="Job description is empty"):
            run_full(b"fake_pdf", "   ")
