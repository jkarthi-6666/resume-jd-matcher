"""Score a single requirement against retrieved evidence chunks."""
import pathlib
from src.schemas import Chunk, Requirement, RequirementAnalysis
from src import llm, config

_PROMPT = (pathlib.Path(__file__).parent.parent / "prompts" / "scorer.txt").read_text()


def score(
    requirement: Requirement,
    chunks: list[Chunk],
    low_retrieval_confidence: bool = False,
    model: str | None = None,
) -> RequirementAnalysis:
    model = model or config.SCORER_MODEL

    if not chunks:
        return RequirementAnalysis(
            requirement_id=requirement.id,
            requirement=requirement.requirement,
            importance=requirement.importance,
            score=0.0,
            confidence=1.0,
            evidence=[],
            reason="No evidence retrieved for this requirement.",
            retrieved_chunk_ids=[],
            low_retrieval_confidence=low_retrieval_confidence,
            evidence_valid=True,
            status="missing",
        )

    chunk_text = "\n\n---\n\n".join(
        f"[chunk_id={c.chunk_id}]\n{c.embed_text}" for c in chunks
    )
    prompt = _PROMPT.format(
        requirement=requirement.requirement,
        importance=requirement.importance,
        retrieved_chunks=chunk_text,
        requirement_id=requirement.id,
    )

    try:
        analysis: RequirementAnalysis = llm.call(
            prompt, model, RequirementAnalysis, temperature=0.0
        )
        # Ensure IDs are correct regardless of what model returned
        analysis.requirement_id = requirement.id
        analysis.requirement   = requirement.requirement
        analysis.importance    = requirement.importance
        analysis.retrieved_chunk_ids = [c.chunk_id for c in chunks]
        analysis.low_retrieval_confidence = low_retrieval_confidence
        return analysis
    except Exception as e:
        # Retry once
        try:
            analysis = llm.call(prompt, model, RequirementAnalysis, temperature=0.0)
            analysis.requirement_id = requirement.id
            analysis.requirement   = requirement.requirement
            analysis.importance    = requirement.importance
            analysis.retrieved_chunk_ids = [c.chunk_id for c in chunks]
            analysis.low_retrieval_confidence = low_retrieval_confidence
            return analysis
        except Exception:
            return RequirementAnalysis(
                requirement_id=requirement.id,
                requirement=requirement.requirement,
                importance=requirement.importance,
                score=0.0,
                confidence=0.0,
                evidence=[],
                reason=f"Scoring failed: {e}",
                retrieved_chunk_ids=[c.chunk_id for c in chunks],
                low_retrieval_confidence=low_retrieval_confidence,
            )
