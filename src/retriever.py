"""Hybrid BM25 + cosine retrieval with reciprocal rank fusion."""
from rank_bm25 import BM25Okapi
from src.schemas import Chunk
from src.embeddings import VectorIndex
from src import config


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
        tokenized = [c.embed_text.lower().split() for c in chunks]
        self._bm25 = BM25Okapi(tokenized)
        self._vector_index.build(chunks)

    def retrieve(
        self,
        query: str,
        bm25_top_k: int | None = None,
        vector_top_k: int | None = None,
    ) -> tuple[list[str], list[str], list[str], float]:
        """
        Returns (bm25_ranked_ids, vector_ranked_ids, rrf_merged_ids, top_rrf_score).
        top_rrf_score is the highest RRF score across merged results.
        """
        bk = bm25_top_k or config.BM25_TOP_K
        vk = vector_top_k or config.VECTOR_TOP_K

        bm25_ranked = self._bm25_query(query, bk)
        vec_ranked  = self._vector_query(query, vk)
        merged      = rrf(bm25_ranked, vec_ranked, k=config.RRF_K)

        top_score = _top_rrf_score(bm25_ranked, vec_ranked, k=config.RRF_K)
        return bm25_ranked, vec_ranked, merged, top_score

    def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        return [self._chunk_map[cid] for cid in chunk_ids if cid in self._chunk_map]

    def _bm25_query(self, query: str, top_k: int) -> list[str]:
        tokens = query.lower().split()
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
