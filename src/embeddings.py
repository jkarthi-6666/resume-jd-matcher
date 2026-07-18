"""FAISS vector index for resume chunks."""
import numpy as np
import faiss
from src.schemas import Chunk
from src.llm import get_embeddings


class VectorIndex:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._index: faiss.IndexFlatIP | None = None

    def build(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build FAISS index: no chunks provided.")
        self._chunks = chunks
        texts = [c.embed_text for c in chunks]
        vecs = get_embeddings(texts, input_type="passage")
        mat = np.array(vecs, dtype="float32")
        faiss.normalize_L2(mat)
        dim = mat.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(mat)

    def query(self, text: str, top_k: int = 8) -> list[tuple[str, float]]:
        """Return list of (chunk_id, score) sorted descending."""
        if self._index is None:
            raise RuntimeError("Index not built. Call build() first.")
        vec = np.array(get_embeddings([text], input_type="query"), dtype="float32")
        faiss.normalize_L2(vec)
        scores, indices = self._index.search(vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append((self._chunks[idx].chunk_id, float(score)))
        return results
