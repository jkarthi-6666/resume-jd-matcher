"""In-memory cosine-similarity vector index for resume chunks."""
import numpy as np
from src.schemas import Chunk
from src.llm import get_embeddings


class VectorIndex:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._index: np.ndarray | None = None

    def build(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build vector index: no chunks provided.")
        self._chunks = chunks
        texts = [c.embed_text for c in chunks]
        vecs = get_embeddings(texts, input_type="passage")
        mat = np.array(vecs, dtype="float32")
        if mat.ndim != 2 or mat.shape[0] != len(chunks) or mat.shape[1] == 0:
            raise ValueError("Embedding provider returned an invalid matrix.")
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        self._index = mat / np.maximum(norms, 1e-12)

    def query(self, text: str, top_k: int = 8) -> list[tuple[str, float]]:
        """Return list of (chunk_id, score) sorted descending."""
        if self._index is None:
            raise RuntimeError("Index not built. Call build() first.")
        vec = np.array(get_embeddings([text], input_type="query"), dtype="float32")
        if vec.ndim != 2 or vec.shape != (1, self._index.shape[1]):
            raise ValueError("Embedding provider returned an invalid query vector.")
        norm = np.linalg.norm(vec[0])
        normalized = vec[0] / max(float(norm), 1e-12)
        scores = self._index @ normalized
        limit = min(max(top_k, 0), len(self._chunks))
        ranked = np.argsort(-scores, kind="stable")[:limit]
        return [
            (self._chunks[int(idx)].chunk_id, float(scores[idx]))
            for idx in ranked
        ]
