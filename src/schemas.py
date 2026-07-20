from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

Importance = Literal["required", "preferred", "unknown"]
# Category is descriptive metadata only; it is not used to score or route a
# candidate.  Some JSON-mode models return ``unknown`` when the JD does not make
# a requirement's category clear.  Treat that as an explicit fallback instead
# of aborting the entire pipeline after requirement extraction.
Category   = Literal["technical_skill", "experience", "education",
                     "domain_knowledge", "soft_skill", "responsibility",
                     "unknown"]
Status     = Literal["matched", "partially_matched", "missing", "uncertain"]
Verdict    = Literal["accept", "needs_review", "reject"]


class Requirement(BaseModel):
    id: str
    requirement: str
    category: Category
    importance: Importance


class RequirementPlan(BaseModel):
    requirements: list[Requirement]


class Chunk(BaseModel):
    chunk_id: str
    section: str
    header: str
    body: str
    embed_text: str
    source: str
    header_detected: bool = True


class RequirementAnalysis(BaseModel):
    # Scores are mutated after construction (validate.py zeroes them, reflector.py
    # applies corrections). Without validate_assignment the ge/le bounds below only
    # guard __init__, so an out-of-range score assigned later would reach the report.
    model_config = ConfigDict(validate_assignment=True)

    requirement_id: str
    requirement: str
    importance: Importance
    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str]
    reason: str
    retrieved_chunk_ids: list[str]
    low_retrieval_confidence: bool = False
    evidence_valid: bool | None = None
    status: Status | None = None


class RerankedChunk(BaseModel):
    chunk_id: str
    justification: str


class RerankerResult(BaseModel):
    ranked_chunks: list[RerankedChunk]


class Correction(BaseModel):
    # Deliberately permissive: this is raw critic output. Constraining scores or
    # cross-checking direction here would fail the whole ReflectionResult on one
    # bad item, discarding every valid correction alongside it. reflector.py
    # enforces the invariants per-correction so the rest of the batch survives.
    requirement_id: str
    old_score: float
    new_score: float
    direction: Literal["raised", "lowered"]
    reason: str
    new_evidence: list[str] = Field(default_factory=list)


class ReflectionResult(BaseModel):
    approved: bool
    changed_requirements: list[Correction]
    review_notes: list[str]
    # Internal reliability signal. LLM responses omit it and therefore default
    # to True; reflector.py sets it False when both audit attempts fail.
    completed: bool = True


class FinalReport(BaseModel):
    verdict: Verdict
    review_reason: str | None = None
    match_score: float
    score_pre_reflection: float
    score_post_reflection: float
    reflection_delta: float
    all_requirements: list[RequirementAnalysis]
    corrections: list[Correction]
    recommendation: str
    reflection_notes: list[str]


# Naive mode output (Phase 1 baseline)
class NaiveResult(BaseModel):
    match_score: int = Field(ge=0, le=100)
    matched_skills: list[str]
    missing_skills: list[str]
    recommendation: str
