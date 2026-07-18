"""LLM-based reranking of retrieved chunks."""
import pathlib
from src.schemas import Chunk, RerankerResult
from src import llm, config

_PROMPT = (pathlib.Path(__file__).parent.parent / "prompts" / "reranker.txt").read_text()


def rerank(
    requirement: str,
    candidates: list[Chunk],
    model: str | None = None,
) -> list[Chunk]:
    """Rerank candidate chunks for a single requirement. Returns top RERANK_TOP_N."""
    if not candidates:
        return []

    model = model or config.RERANKER_MODEL
    numbered = "\n\n".join(
        f"[{i+1}] chunk_id={c.chunk_id}\n{c.embed_text}" for i, c in enumerate(candidates)
    )
    prompt = _PROMPT.format(requirement=requirement, numbered_chunks=numbered)

    try:
        result: RerankerResult = llm.call(prompt, model, RerankerResult, temperature=0.0)
        chunk_map = {c.chunk_id: c for c in candidates}
        reranked = [chunk_map[r.chunk_id] for r in result.ranked_chunks if r.chunk_id in chunk_map]
        # Append any chunks the model dropped (preserve all for debugging)
        seen = {c.chunk_id for c in reranked}
        for c in candidates:
            if c.chunk_id not in seen:
                reranked.append(c)
    except Exception:
        # On failure, return original order
        reranked = candidates

    return reranked[: config.RERANK_TOP_N]
