"""Tests for hybrid retrieval. Key assertion: exact term is always retrieved."""
import pytest
from unittest.mock import patch
from src.schemas import Chunk
from src.retriever import HybridRetriever, rrf, tokenize
from tests.conftest import fake_embeddings


def _make_chunk(chunk_id: str, text: str, section: str = "Experience") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        section=section,
        header="Role at Company 2020-2023",
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


class TestTokenize:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("Python, Java; and Go.", ["python", "java", "and", "go"]),
            ("(Kubernetes)", ["kubernetes"]),
            ("“Flask”", ["flask"]),
        ],
    )
    def test_presentation_punctuation_is_dropped(self, text, expected):
        assert tokenize(text) == expected

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("C++", ["c++"]),
            ("C#", ["c#"]),
            (".NET", [".net"]),
            ("Node.js", ["node.js"]),
            ("scikit-learn", ["scikit-learn"]),
            ("Python 3.11", ["python", "3.11"]),
        ],
    )
    def test_names_carrying_punctuation_survive(self, text, expected):
        """Stripping punctuation wholesale would collapse C++ and C# to "c"."""
        assert tokenize(text) == expected

    def test_slash_separates_alternatives(self):
        assert tokenize("CI/CD on k8s/GKE") == ["ci", "cd", "on", "k8s", "gke"]

    def test_quoted_term_matches_bare_term(self):
        """The bug this fixes: "Python," and "python" were different tokens."""
        assert tokenize("Python,") == tokenize("python")


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

    def test_bm25_matches_a_term_followed_by_punctuation(self):
        """A comma after a skill used to make it a different BM25 token."""
        chunks = [
            _make_chunk("chunk_listed", "Toolchain: Terraform, Ansible, Consul."),
            _make_chunk("chunk_other", "Wrote internal documentation."),
            _make_chunk("chunk_third", "Ran the weekly release meeting."),
        ]
        retriever = HybridRetriever()
        retriever.build(chunks)

        bm25_ranked, _, _, _ = retriever.retrieve("Ansible")

        assert bm25_ranked[0] == "chunk_listed"

    def test_query_terms_expand_bm25_but_not_vector_query(self):
        chunks = [
            _make_chunk("chunk_k8s", "Operated k8s workloads in production."),
            _make_chunk("chunk_java", "Built Java web services."),
            _make_chunk("chunk_sql", "Optimized SQL reporting queries."),
        ]
        retriever = HybridRetriever()
        retriever.build(chunks)

        with patch.object(
            retriever._vector_index,
            "query",
            return_value=[("chunk_java", 0.9)],
        ) as vector_query:
            bm25_ranked, _, _, _ = retriever.retrieve(
                "container orchestration",
                query_terms=["k8s", "Kubernetes"],
            )

        assert bm25_ranked[0] == "chunk_k8s"
        vector_query.assert_called_once_with("container orchestration", 8)
