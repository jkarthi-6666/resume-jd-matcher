"""Hybrid BM25 + cosine retrieval with reciprocal rank fusion."""
import re

from rank_bm25 import BM25Okapi
from src.schemas import Chunk
from src.embeddings import VectorIndex
from src import config

# A word, optionally carrying punctuation that belongs to the name itself:
# internal marks ("node.js", "scikit-learn", "3.11") and trailing ones ("c++",
# "c#"). The second alternative covers leading-dot and hash names (".net",
# "#define") that cannot start with an alphanumeric. A slash is deliberately
# not an internal mark: in resumes it separates alternatives ("CI/CD",
# "k8s/GKE"), so splitting there makes each half individually retrievable.
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[.+#_-][a-z0-9]+)*[+#]*|[.#][a-z][a-z0-9]*")


def tokenize(text: str) -> list[str]:
    """Lowercase and drop punctuation that is only presentation.

    ``str.split()`` alone leaves adjoining punctuation attached, so a resume's
    "Python," never matches a query's "python". Stripping punctuation wholesale
    is worse: it collapses "C++" and "C#" into a bare "c" and erases the only
    token that distinguishes those requirements from each other. Punctuation is
    kept only where it sits inside or at the end of a name.

    Indexing and querying must share this function. A token split one way at
    build time and another way at query time cannot match.
    """
    return _TOKEN_PATTERN.findall(text.lower())


class HybridRetriever:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._chunk_map: dict[str, Chunk] = {}
        self._bm25: BM25Okapi | None = None
        self._vector_index = VectorIndex()

    def build(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build retriever: no chunks provided.")
        self._chunks = chunks
        self._chunk_map = {c.chunk_id: c for c in chunks}
        # BM25 on tokenized embed_text
        tokenized = [tokenize(c.embed_text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized)
        self._vector_index.build(chunks)

    def retrieve(
        self,
        query: str,
        bm25_top_k: int | None = None,
        vector_top_k: int | None = None,
        query_terms: list[str] | None = None,
    ) -> tuple[list[str], list[str], list[str], float]:
        """
        Returns (bm25_ranked_ids, vector_ranked_ids, rrf_merged_ids, top_rrf_score).
        top_rrf_score is the highest RRF score across merged results.
        """
        bk = bm25_top_k or config.BM25_TOP_K
        vk = vector_top_k or config.VECTOR_TOP_K

        bm25_ranked = self._bm25_query(query, bk, query_terms)
        vec_ranked  = self._vector_query(query, vk)
        merged      = rrf(bm25_ranked, vec_ranked, k=config.RRF_K)

        top_score = _top_rrf_score(bm25_ranked, vec_ranked, k=config.RRF_K)
        return bm25_ranked, vec_ranked, merged, top_score

    def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        return [self._chunk_map[cid] for cid in chunk_ids if cid in self._chunk_map]

    def _bm25_query(
        self,
        query: str,
        top_k: int,
        query_terms: list[str] | None = None,
    ) -> list[str]:
        tokens = tokenize(query)
        tokens.extend(
            token
            for term in (query_terms or [])
            for token in tokenize(term)
        )
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(self._chunks)), key=lambda i: scores[i], reverse=True)
        return [self._chunks[i].chunk_id for i in ranked[:top_k]]

    def _vector_query(self, query: str, top_k: int) -> list[str]:
        results = self._vector_index.query(query, top_k)
        return [cid for cid, _ in results]


def rrf(bm25_ranked: list[str], vec_ranked: list[str], k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in (bm25_ranked, vec_ranked):
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda cid: scores[cid], reverse=True)


def _top_rrf_score(bm25_ranked: list[str], vec_ranked: list[str], k: int) -> float:
    scores: dict[str, float] = {}
    for ranking in (bm25_ranked, vec_ranked):
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return max(scores.values()) if scores else 0.0
