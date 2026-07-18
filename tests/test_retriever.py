"""Tests for hybrid retrieval. Key assertion: exact term is always retrieved."""
import pytest
from unittest.mock import patch
from src.schemas import Chunk
from src.retriever import HybridRetriever, rrf
from tests.conftest import fake_embeddings


def _make_chunk(chunk_id: str, text: str, section: str = "Experience") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        section=section,
        header=f"Role at Company 2020-2023",
        body=text,
        embed_text=text,
        source="test.pdf",
    )


CHUNKS = [
    _make_chunk("chunk_001", "Led a Kubernetes migration across four teams. Managed GKE clusters."),
    _make_chunk("chunk_002", "Built Flask REST APIs for invoice processing and email extraction."),
    _make_chunk("chunk_003", "Python, Docker, Kubernetes, Terraform, AWS", section="Skills"),
    _make_chunk("chunk_004", "Managed PostgreSQL and Redis databases."),
    _make_chunk("chunk_005", "B.S. Computer Science, MIT, 2018."),
]


class TestRRF:
    def test_rrf_merges_rankings(self):
        bm25 = ["chunk_001", "chunk_003", "chunk_002"]
        vec  = ["chunk_002", "chunk_001", "chunk_004"]
        merged = rrf(bm25, vec)
        assert "chunk_001" in merged[:3]

    def test_rrf_unique_results(self):
        bm25 = ["chunk_001", "chunk_002", "chunk_003"]
        vec  = ["chunk_001", "chunk_003", "chunk_002"]
        merged = rrf(bm25, vec)
        assert len(merged) == len(set(merged))

    def test_rrf_first_ranked_in_both_wins(self):
        bm25 = ["chunk_A", "chunk_B"]
        vec  = ["chunk_A", "chunk_C"]
        merged = rrf(bm25, vec)
        assert merged[0] == "chunk_A"


class TestHybridRetriever:
    @pytest.fixture(autouse=True)
    def patch_embeddings(self):
        """Avoid real API calls.

        src.embeddings is the only module that calls get_embeddings; retriever
        reaches it via VectorIndex, so there is nothing to patch on retriever
        itself.
        """
        with patch("src.embeddings.get_embeddings", side_effect=fake_embeddings):
            yield

    def test_exact_term_retrieved_by_bm25(self):
        """If the resume contains an exact term, the chunk must be in the results."""
        retriever = HybridRetriever()
        retriever.build(CHUNKS)
        bm25_ranked, _, merged, _ = retriever.retrieve("Kubernetes experience")
        # chunk_001 and chunk_003 both contain "Kubernetes" — at least one must be top-4
        assert any(cid in ("chunk_001", "chunk_003") for cid in bm25_ranked[:4])

    def test_retrieve_returns_merged_list(self):
        retriever = HybridRetriever()
        retriever.build(CHUNKS)
        bm25_r, vec_r, merged, top_score = retriever.retrieve("Python REST APIs")
        assert isinstance(merged, list)
        assert len(merged) > 0

    def test_top_rrf_score_positive(self):
        retriever = HybridRetriever()
        retriever.build(CHUNKS)
        _, _, _, top_score = retriever.retrieve("PostgreSQL database")
        assert top_score > 0.0

    def test_get_chunks_returns_correct(self):
        retriever = HybridRetriever()
        retriever.build(CHUNKS)
        chunks = retriever.get_chunks(["chunk_001", "chunk_003"])
        ids = [c.chunk_id for c in chunks]
        assert "chunk_001" in ids
        assert "chunk_003" in ids
